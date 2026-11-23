//! A synthetic 10-minute broadcast: a logo in the top-right corner during the programme, black-and-silent edges
//! around ad breaks, a break that ends without one (a split-screen return), and a logo-less replay with no edges.

use adbreak::{Config, Detector, Frame};

const W: usize = 480;
const H: usize = 270;
const FPS: f64 = 2.0;
const RATE: u32 = 8_000;

#[derive(Clone, Copy, PartialEq)]
enum Shot {
    Programme,
    Ad,
    Black,
}

fn frame(shot: Shot, t: f64) -> Vec<u8> {
    let mut y = vec![0u8; W * H];
    if shot == Shot::Black {
        y.fill(16);
        return y;
    }
    // Moving content: diagonal stripes that drift, so corners change from second to second.
    let shift = (t * 7.0) as usize;
    for (i, p) in y.iter_mut().enumerate() {
        let (r, c) = (i / W, i % W);
        *p = if ((r + c + shift) / 23).is_multiple_of(2) { 70 } else { 130 };
    }
    if shot == Shot::Programme {
        // The logo: a bright rectangle outline inside the top-right corner box.
        let (x0, y0, x1, y1) = (430, 9, 470, 22);
        for r in y0..=y1 {
            for c in x0..=x1 {
                if r == y0 || r == y1 || c == x0 || c == x1 || r == y0 + 1 || r == y1 - 1 || c == x0 + 1 || c == x1 - 1 {
                    y[r * W + c] = 235;
                }
            }
        }
    }
    y
}

/// What is on screen at `t`, and whether the sound is silent.
fn script(t: f64, with_silence: bool) -> (Shot, bool) {
    let edge = |at: f64| t >= at && t < at + 0.5;
    for at in [120.0, 240.0, 400.0] {
        if edge(at) {
            return (Shot::Black, with_silence);
        }
    }
    let shot = if (120.0..240.0).contains(&t) || (400.0..520.0).contains(&t) || (300.0..330.0).contains(&t) { Shot::Ad } else { Shot::Programme };
    (shot, false)
}

fn run(with_silence: bool, logo: bool) -> Vec<adbreak::Break> {
    let mut d = Detector::new(Config::default());
    let mut t = 0.0;
    while t < 600.0 {
        let (mut shot, silent) = script(t, with_silence);
        if !logo && shot == Shot::Programme {
            shot = Shot::Ad;
        }
        let y = frame(shot, t);
        d.push_frame(Frame { pts: t, width: W, height: H, luma: &y });
        let n = (RATE as f64 / FPS) as usize;
        let audio: Vec<f32> = (0..n).map(|i| if silent { 0.0 } else { 0.2 * ((t * RATE as f64 + i as f64) * 0.05).sin() as f32 }).collect();
        d.push_audio(t, RATE, &audio);
        t += 1.0 / FPS;
    }
    d.finish();
    d.breaks()
}

#[test]
fn finds_both_breaks_and_ignores_the_replay() {
    let b = run(true, true);
    assert_eq!(b.len(), 2, "{b:?}");
    // Ends at the black-and-silent edge where the programme comes back.
    assert!((b[0].start - 120.25).abs() < 0.5 && (b[0].end - 240.25).abs() < 0.5, "{b:?}");
    // Ends where the logo comes back, with no edge there (split-screen return).
    assert!((b[1].start - 400.25).abs() < 0.5 && (b[1].end - 520.0).abs() < 1.0, "{b:?}");
}

#[test]
fn black_without_silence_is_not_an_edge() {
    assert!(run(false, true).is_empty());
}

#[test]
fn without_a_logo_nothing_can_be_told_apart() {
    assert!(run(true, false).is_empty());
}
