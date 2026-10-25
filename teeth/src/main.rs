// SPDX-License-Identifier: MPL-2.0
// Copyright 2026 Vyges.
//! Break one rule, name the test that must notice.
//!
//! ⛔ **A test that cannot FAIL proves nothing.** Two of `pad`'s first order probes were inert
//! when written and nobody noticed until the runner was taught to check. This is that check: each
//! entry breaks ONE rule and names the test that must catch it.
//!
//! # Why this is Rust and not a shell script
//!
//! The shell version's every defect came from the shell, and each one produced a **green result
//! over work that never happened**:
//!
//! * the table was a list of bash double-quoted strings, so a raw `"` inside a pattern ended its
//!   row early and word-split the remainder into separate arguments. A 183-row sweep silently ran
//!   **106** and still printed `0 holes`. ⟹ Here the table is a typed array of [`Mutation`] and a
//!   stray quote is a **compile error**.
//! * the verdict came from grepping `cargo test`'s log. An aborted test binary truncates that log
//!   at a point decided by thread scheduling, so the *same* mutation classified as `caught` in one
//!   run and `WRONG TEST` in the next. ⟹ Here the verdict is the **exit code of one named test**,
//!   run alone. Nothing is parsed to decide anything.
//! * a mutation whose source did not compile went red and read as covered. ⟹ Here compilation is
//!   a separate step with its own outcome.
//! * a `kill -9` skipped the restore trap and left a mutated file looking like ordinary
//!   uncommitted work. ⟹ Here the restore is a [`Drop`] guard, and a repair pass runs at startup
//!   for the run that was killed before its own guard could fire.

use std::collections::BTreeSet;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

mod mutations;
use mutations::MUTATIONS;

/// One broken rule, and the test that must notice.
///
/// ⚠️ `find` is matched **literally**, once, at its first occurrence — not as a regex. Anchor on
/// a fragment that is unique in the file; it does not have to be complete.
pub struct Mutation {
    pub name: &'static str,
    pub file: &'static str,
    pub find: &'static str,
    pub replace: &'static str,
    /// The test that must fail. It is run **alone**, so this must name it exactly.
    pub want: &'static str,
}

/// The four things that can happen, deliberately distinct — they mean different things.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Outcome {
    /// The named test failed. The rule is pinned.
    Caught,
    /// The suite went red, but the named test passed. The rule is covered by SOMETHING and our
    /// belief about which test covers it was wrong. Fix the expectation.
    WrongTest,
    /// The suite stayed green. A real hole.
    NotCaught,
    /// The pattern did not match, so nothing was measured.
    StalePattern,
    /// The `want` names a test that does not exist, so nothing was measured. ⛔ **Kept apart from
    /// [`Outcome::StalePattern`] deliberately.** The two were one bucket until 2026-08-26, and the
    /// conflation sent a reader looking for a broken `find` pattern when the pattern was fine and
    /// the TEST had been renamed. A rule with no test and a rule with no site are different
    /// repairs.
    NoSuchTest,
    /// The mutated source does not build, so nothing was measured. The stale pattern's twin:
    /// red, and proving nothing.
    DoesNotCompile,
}

fn main() -> ExitCode {
    let root = match crate_root() {
        Some(r) => r,
        None => {
            eprintln!("ERROR: run this from the vyges-sparselu checkout (no Cargo.toml above me)");
            return ExitCode::from(2);
        }
    };

    // ⚠️ A `kill -9` skips the Drop guard. The run that was killed cannot repair itself; the next
    // one can, and what it leaves behind otherwise looks exactly like ordinary uncommitted work.
    let repaired = repair_leftovers(&root);
    for f in &repaired {
        eprintln!("note: restored {} left mutated by an earlier run", f.display());
    }

    let arg: Option<String> = std::env::args().nth(1);
    if arg.as_deref() == Some("--batches") {
        list_batches();
        return ExitCode::SUCCESS;
    }
    let selected: Vec<&Mutation> = match &arg {
        None => MUTATIONS.iter().collect(),
        Some(a) => select(a),
    };

    // ⛔ A filter that selects nothing prints `0 caught, 0 problems`, which reads exactly like a
    // pass. It is not one.
    if selected.is_empty() {
        eprintln!("ERROR: no mutation matches {:?} — nothing was measured", arg.unwrap());
        eprintln!("       batches: {}", BATCHES.iter().map(|b| b.0).collect::<Vec<_>>().join(", "));
        return ExitCode::from(2);
    }
    if let Err(e) = check_names_unique() {
        eprintln!("ERROR: {e}");
        return ExitCode::from(2);
    }

    // 🔑 **Run in batches, and summarise each one as it lands.** A 210-line wall is not a result
    // anyone reads; three sections with their own verdict are. The batches are derived from the
    // file each mutation targets, so a new mutation joins the right one without being told.
    let mut total = [0usize; 6];
    let mut ran = 0usize;
    for (batch, _) in BATCHES {
        let group: Vec<&&Mutation> = selected.iter().filter(|m| batch_of(m) == *batch).collect();
        if group.is_empty() {
            continue;
        }
        println!("\n\x1b[1m{batch}\x1b[0m ({} mutations)", group.len());
        let mut counts = [0usize; 6];
        for m in group {
            let outcome = run_one(&root, m);
            report(m, outcome);
            counts[outcome as usize] += 1;
            total[outcome as usize] += 1;
            ran += 1;
        }
        println!("  \u{2514}\u{2500} {}", summary(&counts));
        let _ = std::io::stdout().flush();
    }

    println!("\nteeth: {} of {ran}", summary(&total));
    let counts = total;
    assert_eq!(ran, selected.len(), "every selected mutation belongs to exactly one batch");

    // ⚠️ Verify the tree really is back to green, rather than trusting that every restore ran.
    if !cargo(&root, &["test", "--offline"]).success {
        eprintln!("ERROR: the suite is not green after restoring — the tree may still be mutated");
        return ExitCode::from(2);
    }
    if counts[Outcome::Caught as usize] == selected.len() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn run_one(root: &Path, m: &Mutation) -> Outcome {
    let path = root.join(m.file);
    let original = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("  {:<34} cannot read {}: {e}", m.name, path.display());
            return Outcome::StalePattern;
        }
    };
    let Some(at) = original.find(m.find) else {
        return Outcome::StalePattern;
    };
    let mut mutated = String::with_capacity(original.len());
    mutated.push_str(&original[..at]);
    mutated.push_str(m.replace);
    mutated.push_str(&original[at + m.find.len()..]);
    if mutated == original {
        return Outcome::StalePattern;
    }

    // ⛔ **The backup goes to DISK before the file is touched, not just into memory.** `Drop`
    // does not run for SIGTERM or SIGKILL, so a killed run cannot restore itself — and an
    // in-memory copy dies with the process, leaving a mutated source and NOTHING to repair from.
    // Measured the hard way on 2026-08-25: a `pkill` left `src/shaping.rs` with rows and columns
    // swapped, looking exactly like ordinary uncommitted work.
    let backup = path.with_extension(format!("{}.teeth-backup", extension_of(&path)));
    if let Err(e) = std::fs::write(&backup, &original) {
        eprintln!("  {:<34} cannot write {}: {e}", m.name, backup.display());
        return Outcome::StalePattern;
    }
    let _guard = Restore { path: path.clone(), backup };
    if std::fs::write(&path, &mutated).is_err() {
        return Outcome::StalePattern;
    }

    // Step 1: does it build? Answered by its own exit code, not by reading a log.
    //
    // ⚠️ Only the ONE target that holds the named test. `--tests` builds all eighteen of them and
    // turned a 90-second sweep into a twelve-minute one.
    let target = test_target(root, m.want);
    let mut build = vec!["build", "--offline"];
    match &target {
        Where::Integration(_) | Where::Lib => build.extend(target.argv()),
        Where::Unknown => build.push("--tests"),
    }
    if !cargo(root, &build).success {
        return Outcome::DoesNotCompile;
    }

    // Step 2: the verdict — run the ONE named test, alone.
    //
    // 🔑 Alone is what makes this deterministic. In a batch, a different test can overflow its
    // stack and abort the whole binary before the named one reports, and which tests got that far
    // depends on thread scheduling.
    // ⚠️ A unit test inside `mod tests` is addressed by its FULL path — `status::tests::name` —
    // and `--exact` on the bare name matches nothing at all. The path is discovered from
    // `--list` rather than assumed from the file name, because the module nesting is the
    // author's choice, not a convention this harness gets to impose.
    let exact = match &target {
        Where::Lib => match lib_test_path(root, m.want) {
            Some(path) => path,
            None => {
                return Outcome::NoSuchTest;
            }
        },
        _ => m.want.to_string(),
    };
    let mut argv = vec!["test", "--offline"];
    argv.extend(target.argv());
    argv.extend(["--", "--exact", &exact]);
    let solo = cargo(root, &argv);

    // 🔑 **Order matters here.** A run that failed is a verdict; a run that succeeded might not
    // have run anything at all. Taking them the other way round misreads a CRASHING test as a
    // missing one: a mutation that removes a recursion guard makes the binary overflow its stack
    // and abort before libtest prints a single line, and the name is then nowhere in the output.
    if !solo.success {
        return Outcome::Caught;
    }

    // ⛔ `--exact` on a name that matches nothing runs ZERO tests and exits 0 — which would read
    // as NOT CAUGHT: a hole reported where there is only a typo. Safe to test only now, because
    // the run above succeeded, so the absence of the name really does mean absence.
    //
    // ⚠️ The check is for the named test's OWN line, not for `running 0 tests`. With several test
    // binaries, every binary that does not hold the test prints `running 0 tests`, so that string
    // is present on almost every successful run — using it reported ten live mutations as stale.
    //
    // ℹ️ This is the one place the log is read, and it decides only whether the test EXISTS. The
    // verdict itself is the exit code.
    if !solo.stdout.contains(&format!("{} ...", m.want)) {
        return Outcome::NoSuchTest;
    }

    // Step 3: the named test passed. Did anything else notice?
    if cargo(root, &["test", "--offline"]).success {
        Outcome::NotCaught
    } else {
        Outcome::WrongTest
    }
}

/// Restores the file when this value is dropped — on return, on `?`, and on panic.
///
/// ⚠️ **Not on a signal.** `Drop` never runs for SIGTERM or SIGKILL, which is why the backup is a
/// file on disk: what this cannot repair, the next run's startup pass can.
struct Restore {
    path: PathBuf,
    backup: PathBuf,
}

impl Drop for Restore {
    fn drop(&mut self) {
        if let Err(e) = std::fs::rename(&self.backup, &self.path) {
            // ⛔ Loud: a failed restore leaves a mutated source in the working tree.
            eprintln!("ERROR: could not restore {}: {e}", self.path.display());
            return;
        }
        // ⚠️ `rename` carries the backup's older mtime across, and cargo would then keep the
        // binary it built from the MUTATED source — measuring the next mutation against the
        // previous one. Touch it so the rebuild actually happens.
        touch(&self.path);
    }
}

fn touch(path: &Path) {
    // Rewriting the file in place is the portable way to move its mtime forward.
    if let Ok(contents) = std::fs::read(path) {
        let _ = std::fs::write(path, contents);
    }
}

fn extension_of(path: &Path) -> String {
    path.extension().map(|e| e.to_string_lossy().into_owned()).unwrap_or_default()
}

struct Run {
    success: bool,
    stdout: String,
}

fn cargo(root: &Path, args: &[&str]) -> Run {
    match Command::new(cargo_bin()).args(args).current_dir(root).output() {
        Ok(o) => Run {
            success: o.status.success(),
            stdout: String::from_utf8_lossy(&o.stdout).into_owned(),
        },
        Err(e) => {
            eprintln!("ERROR: could not run cargo: {e}");
            Run { success: false, stdout: String::new() }
        }
    }
}

fn cargo_bin() -> String {
    std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string())
}

/// Where a test lives, so only that target is built and run.
enum Where {
    /// `tests/<name>.rs`.
    Integration(String),
    /// A `#[test]` inside `src/` — reached with `--lib`, NOT with `--test`.
    Lib,
    /// Could not tell; the whole suite is used, which is the safe default.
    Unknown,
}

impl Where {
    fn argv(&self) -> Vec<&str> {
        match self {
            Where::Integration(name) => vec!["--test", name],
            Where::Lib => vec!["--lib"],
            Where::Unknown => vec![],
        }
    }
}

/// Find `want`, preferring an integration test and falling back to the library.
///
/// ⚠️ **The library case is not optional.** Omitting it was a regression in the port from the
/// shell version: ten mutations name unit tests that live in `src/`, and without `--lib` the run
/// spreads over every target and the exact-name check cannot tell "absent" from "elsewhere".
fn test_target(root: &Path, want: &str) -> Where {
    let needle = format!("fn {want}(");
    let mut hits: Vec<String> = Vec::new();
    if let Ok(dir) = std::fs::read_dir(root.join("tests")) {
        for entry in dir.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                continue;
            }
            if std::fs::read_to_string(&path).is_ok_and(|s| s.contains(&needle)) {
                if let Some(stem) = path.file_stem() {
                    hits.push(stem.to_string_lossy().into_owned());
                }
            }
        }
    }
    // ⚠️ A name in two files makes `--exact` ambiguous; fall back to the whole suite rather than
    // picking whichever the directory listed first.
    if hits.len() == 1 {
        return Where::Integration(hits.remove(0));
    }
    if hits.is_empty() && src_contains(&root.join("src"), &needle) {
        return Where::Lib;
    }
    Where::Unknown
}

/// The full `module::path::name` of a library test, from `cargo test --lib -- --list`.
///
/// ℹ️ `--list` prints one `path: test` line per test, so this is an exact suffix match, not a
/// guess at how the modules nest.
fn lib_test_path(root: &Path, want: &str) -> Option<String> {
    let listed = cargo(root, &["test", "--offline", "--lib", "--", "--list"]);
    let suffix = format!("::{want}");
    let mut hits = listed
        .stdout
        .lines()
        .filter_map(|l| l.strip_suffix(": test"))
        .filter(|p| *p == want || p.ends_with(&suffix));
    let first = hits.next()?.to_string();
    // ⚠️ Two tests with the same leaf name in different modules make `--exact` a coin toss.
    hits.next().is_none().then_some(first)
}

fn src_contains(dir: &Path, needle: &str) -> bool {
    let Ok(entries) = std::fs::read_dir(dir) else { return false };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if src_contains(&path, needle) {
                return true;
            }
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs")
            && std::fs::read_to_string(&path).is_ok_and(|s| s.contains(needle))
        {
            return true;
        }
    }
    false
}

fn repair_leftovers(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for dir in ["src", "tests"] {
        let Ok(entries) = std::fs::read_dir(root.join(dir)) else { continue };
        for entry in entries.flatten() {
            let backup = entry.path();
            if backup.extension().and_then(|e| e.to_str()) != Some("teeth-backup") {
                continue;
            }
            let target = backup.with_extension("");
            if std::fs::rename(&backup, &target).is_ok() {
                touch(&target);
                out.push(target);
            }
        }
    }
    out
}

/// ⚠️ Two rows with one name make a filtered run ambiguous and a report unreadable.
fn check_names_unique() -> Result<(), String> {
    let mut seen = BTreeSet::new();
    for m in MUTATIONS {
        if !seen.insert(m.name) {
            return Err(format!("two mutations are both named {:?}", m.name));
        }
    }
    Ok(())
}

fn crate_root() -> Option<PathBuf> {
    let mut dir = std::env::current_dir().ok()?;
    loop {
        if dir.join("Cargo.toml").is_file() && dir.join("src").join("lib.rs").is_file() {
            return Some(dir);
        }
        if !dir.pop() {
            return None;
        }
    }
}

/// The batches, in the order the factorisation runs: the ordering, the factorisation's stages,
/// the dense kernels they call, the solves. ⚠️ Every `src/*.rs` a mutation targets must appear in
/// exactly one batch — [`batch_of`] fails loudly rather than silently dropping a file nobody listed.
const BATCHES: &[(&str, &[&str])] = &[
    ("ordering", &["colamd.rs", "etree.rs", "relax.rs"]),
    ("factorisation", &["factor.rs"]),
    ("kernels", &["dense.rs"]),
    ("solve", &["lib.rs"]),
];

fn batch_of(m: &Mutation) -> &'static str {
    let file = m.file.rsplit('/').next().unwrap_or(m.file);
    for (batch, files) in BATCHES {
        if files.contains(&file) {
            return batch;
        }
    }
    // ⛔ Not a default: a file nobody listed would vanish from every batch and from the count.
    panic!("{} is in no batch — add it to BATCHES", m.file)
}

/// A batch name, or a substring of a mutation name or of its file.
fn select(arg: &str) -> Vec<&'static Mutation> {
    if BATCHES.iter().any(|(b, _)| *b == arg) {
        return MUTATIONS.iter().filter(|m| batch_of(m) == arg).collect();
    }
    MUTATIONS.iter().filter(|m| m.name.contains(arg) || m.file.contains(arg)).collect()
}

fn list_batches() {
    for (batch, _) in BATCHES {
        let n = MUTATIONS.iter().filter(|m| batch_of(m) == *batch).count();
        println!("{batch:<12} {n:>3} mutations");
    }
    println!("{:<12} {:>3} mutations", "(total)", MUTATIONS.len());
}

fn summary(c: &[usize; 6]) -> String {
    format!(
        "{} caught, {} wrong-test, {} holes, {} stale, {} no-such-test, {} uncompilable",
        c[Outcome::Caught as usize],
        c[Outcome::WrongTest as usize],
        c[Outcome::NotCaught as usize],
        c[Outcome::StalePattern as usize],
        c[Outcome::NoSuchTest as usize],
        c[Outcome::DoesNotCompile as usize],
    )
}

fn report(m: &Mutation, outcome: Outcome) {
    let (colour, label, tail) = match outcome {
        Outcome::Caught => ("32", "caught", format!(" by {}", m.want)),
        Outcome::WrongTest => ("33", "WRONG TEST", format!(", but {} passed", m.want)),
        Outcome::NotCaught => ("31", "NOT CAUGHT", " -- the suite stayed green".into()),
        Outcome::StalePattern => ("33", "STALE PATTERN", " (it did not apply)".into()),
        Outcome::NoSuchTest => {
            ("33", "NO SUCH TEST", format!(" -- nothing is named {}", m.want))
        }
        Outcome::DoesNotCompile => ("35", "DOES NOT COMPILE", " -- nothing was measured".into()),
    };
    println!("  {:<34} \x1b[{colour}m{label}\x1b[0m{tail}", m.name);
    // ⚠️ stdout is block-buffered when it is a FILE rather than a terminal, so a redirected run
    // shows nothing until it ends — and a run that is killed shows nothing at all. A sweep is
    // long enough that watching it matters, so every verdict is flushed as it is decided.
    let _ = std::io::stdout().flush();
}
