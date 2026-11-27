//! adbreak's Rust code for its Python training (`ml/`), as the module `adbreak_rs`: the pieces training and a recorder
//! must compute identically (the trees' input table, breaks from probabilities, combining the kinds, caption parsing,
//! text tokens and scores, the model version), so there is one implementation rather than a Python copy kept in step.
//! Learning (scikit-learn) stays in Python.

use adbreak_model::{Columns, text};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use std::collections::BTreeSet;

fn vec(a: &PyReadonlyArray1<f64>) -> PyResult<Vec<f64>> {
    Ok(a.as_slice()?.to_vec())
}

/// The trees' input table (`adbreak_model::table`): seconds × columns, float64 (each value rounded to float32).
#[pyfunction]
#[pyo3(signature = (p, has_captions, visible, roll_up, new_cue, black, silent, edges, windows, edge_cap, text_now=None, text_trail=None))]
#[allow(clippy::too_many_arguments)]
fn table<'py>(
    py: Python<'py>,
    p: PyReadonlyArray1<f64>,
    has_captions: PyReadonlyArray1<f64>,
    visible: PyReadonlyArray1<f64>,
    roll_up: PyReadonlyArray1<f64>,
    new_cue: PyReadonlyArray1<f64>,
    black: PyReadonlyArray1<f64>,
    silent: PyReadonlyArray1<f64>,
    edges: Vec<f64>,
    windows: Vec<usize>,
    edge_cap: f64,
    text_now: Option<PyReadonlyArray1<f64>>,
    text_trail: Option<PyReadonlyArray1<f64>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let p = vec(&p)?;
    let n = p.len();
    let c = Columns {
        has_captions: vec(&has_captions)?,
        visible: vec(&visible)?,
        roll_up: vec(&roll_up)?,
        new_cue: vec(&new_cue)?,
        black: vec(&black)?,
        silent: vec(&silent)?,
    };
    for (name, col) in [
        ("has_captions", &c.has_captions),
        ("visible", &c.visible),
        ("roll_up", &c.roll_up),
        ("new_cue", &c.new_cue),
        ("black", &c.black),
        ("silent", &c.silent),
    ] {
        if col.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("{name} has {} seconds, p has {n}", col.len())));
        }
    }
    let text = match (text_now, text_trail) {
        (Some(a), Some(b)) => Some((vec(&a)?, vec(&b)?)),
        (None, None) => None,
        _ => return Err(pyo3::exceptions::PyValueError::new_err("give both text scores or neither")),
    };
    let rows = py.detach(|| adbreak_model::table(&c, &edges, &p, &windows, edge_cap, text.as_ref().map(|(a, b)| (a.as_slice(), b.as_slice()))));
    let width = rows.first().map_or(0, Vec::len);
    let flat: Vec<f64> = rows.into_iter().flatten().collect();
    Ok(numpy::ndarray::Array2::from_shape_vec((n, width), flat).expect("rows of equal width").into_pyarray(py))
}

/// Mean over ±`half` seconds, clipped at the ends (`adbreak_model::window_mean`).
#[pyfunction]
fn window_mean<'py>(py: Python<'py>, x: PyReadonlyArray1<f64>, half: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(adbreak_model::window_mean(x.as_slice()?, half).into_pyarray(py))
}

/// The middle of each run of seconds with any black (`adbreak_model::black_points`).
#[pyfunction]
fn black_points(black: PyReadonlyArray1<f64>) -> PyResult<Vec<f64>> {
    Ok(adbreak_model::black_points_of(black.as_slice()?))
}

/// Per-second probabilities to breaks (`adbreak_model::to_breaks_snapping`): (start, end) pairs.
#[pyfunction]
#[pyo3(signature = (p, edges, threshold, min_len, max_gap, snap, fallback=Vec::new()))]
fn to_breaks(
    p: PyReadonlyArray1<f64>,
    edges: Vec<f64>,
    threshold: f64,
    min_len: f64,
    max_gap: f64,
    snap: f64,
    fallback: Vec<f64>,
) -> PyResult<Vec<(f64, f64)>> {
    Ok(adbreak_model::to_breaks_snapping(p.as_slice()?, &edges, &fallback, threshold, min_len, max_gap, snap).into_iter().map(|b| (b.start, b.end)).collect())
}

/// Commercial breaks and the other kinds' runs as one list (`adbreak_model::combine`): `runs` maps a kind to its runs
/// and its `min_len`. (start, end, kind) in time order.
#[pyfunction]
fn combine(ad: Vec<(f64, f64)>, runs: adbreak_model::KindRuns, ad_min: f64) -> Vec<(f64, f64, String)> {
    adbreak_model::combine(ad, &runs, ad_min).into_iter().map(|b| (b.start, b.end, b.kind)).collect()
}

/// WebVTT cues (`adbreak_captions::parse_vtt`): (start, end, text), the text its lines joined with a space.
#[pyfunction]
fn parse_vtt(text: &str) -> Vec<(f64, f64, String)> {
    adbreak_captions::parse_vtt(text).into_iter().map(|c| (c.start, c.end, c.lines.join(" "))).collect()
}

/// A caption's token buckets (`adbreak_model::text::tokens`), sorted.
#[pyfunction]
fn tokens(text: &str, buckets: usize) -> Vec<usize> {
    text::tokens(text, buckets).into_iter().collect()
}

/// A sparse matrix's rows: (indptr, indices).
type Csr = (Vec<i64>, Vec<i64>);

fn csr(sets: &[BTreeSet<usize>]) -> Csr {
    let mut indptr = vec![0i64];
    let mut indices = vec![];
    for s in sets {
        indices.extend(s.iter().map(|&j| j as i64));
        indptr.push(indices.len() as i64);
    }
    (indptr, indices)
}

/// Per second 0..n, the tokens on screen and those of cues started in the last `trail` seconds
/// (`adbreak_model::text::per_second`), each as CSR (indptr, indices), indices sorted within a row.
#[pyfunction]
fn text_per_second(cues: Vec<(f64, f64, String)>, n: usize, buckets: usize, trail: usize) -> (Csr, Csr) {
    let (now, trail) = text::per_second(&cues, n, buckets, trail);
    (csr(&now), csr(&trail))
}

/// A text score from exported weights (`adbreak_model::text::score_with`), per row of a CSR token matrix.
#[pyfunction]
fn text_score<'py>(py: Python<'py>, indptr: Vec<i64>, indices: Vec<i64>, coef: Vec<f32>, intercept: f64) -> Bound<'py, PyArray1<f64>> {
    let sets: Vec<BTreeSet<usize>> = indptr.windows(2).map(|w| indices[w[0] as usize..w[1] as usize].iter().map(|&j| j as usize).collect()).collect();
    text::score_with(&sets, &coef, intercept).into_pyarray(py)
}

/// A model.json's version (`adbreak_model::version`).
#[pyfunction]
fn version(model_json: &str) -> String {
    adbreak_model::version(model_json)
}

/// The tree input names (`adbreak_model::feature_names_with`).
#[pyfunction]
fn feature_names(windows: Vec<usize>, text: bool) -> Vec<String> {
    adbreak_model::feature_names_with(&windows, text)
}

#[pymodule]
fn adbreak_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [
        wrap_pyfunction!(table, m)?,
        wrap_pyfunction!(window_mean, m)?,
        wrap_pyfunction!(black_points, m)?,
        wrap_pyfunction!(to_breaks, m)?,
        wrap_pyfunction!(combine, m)?,
        wrap_pyfunction!(parse_vtt, m)?,
        wrap_pyfunction!(tokens, m)?,
        wrap_pyfunction!(text_per_second, m)?,
        wrap_pyfunction!(text_score, m)?,
        wrap_pyfunction!(version, m)?,
        wrap_pyfunction!(feature_names, m)?,
    ] {
        m.add_function(f)?;
    }
    Ok(())
}
