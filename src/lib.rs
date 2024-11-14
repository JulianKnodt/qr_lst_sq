/// Floating point type.
/// Can be changed to f64 with `--feature f64`
#[cfg(not(feature = "f64"))]
pub type F = f32;

/// Floating point type.
/// Can be changed to f64 with `--feature f64`
#[cfg(feature = "f64")]
pub type F = f64;

use std::array::from_fn;

type Matrix<const R: usize, const C: usize> = [[F; C]; R];

fn dot<const N: usize>(a: [F; N], b: [F; N]) -> F {
    (0..N).map(|i| a[i] * b[i]).sum::<F>()
}

fn norm<const N: usize>(v: [F; N]) -> F {
    dot(v, v).max(0.).sqrt()
}

fn norm_iter(v: impl IntoIterator<Item = F>) -> F {
    v.into_iter().map(|v| v * v).sum::<F>().max(0.).sqrt()
}

fn sub<const N: usize>(a: [F; N], b: [F; N]) -> [F; N] {
    from_fn(|i| a[i] - b[i])
}
fn kmul<const N: usize>(k: F, a: [F; N]) -> [F; N] {
    a.map(|v| v * k)
}

/// Transpose a 2D array [[F; C]; R] -> [[F; R]; C].
pub fn transpose<const M: usize, const N: usize>(a: Matrix<M, N>) -> Matrix<N, M> {
    from_fn(|i| from_fn(|j| a[j][i]))
}

/// Decompose A: [[F; C]; R] into (Q,R): ([[F; C]; R], [[F; C];C]), where A = QR.
/// `mgs` stands for modified gram schmidt, which is the algorithm for QR decomposition.
pub fn mgs_qr<const R: usize, const C: usize>(a: Matrix<R, C>) -> (Matrix<R, C>, Matrix<C, C>) {
    let mut q = [[0.; C]; R];
    let mut r = [[0.; C]; C];

    // transpose for later convenience
    let mut vs: Matrix<C, R> = transpose(a);

    for i in 0..C {
        let v: [_; R] = vs[i];
        r[i][i] = norm(v);
        // no need for abs here since it should always be positive
        if r[i][i] < F::EPSILON {
            continue;
        }
        let irii = r[i][i].recip();
        for j in 0..R {
            q[j][i] = irii * v[j];
        }
        let q_i: [_; R] = from_fn(|j| q[j][i]);

        // current column
        for j in i..C {
            r[i][j] = dot(q_i, vs[j]);
            vs[j] = sub(vs[j], kmul(r[i][j], q_i));
        }
    }

    (q, r)
}

fn upper_right_triangular_solve<const N: usize>(u: Matrix<N, N>, b: [F; N]) -> [F; N] {
    let mut out = [0.; N];
    let rcond = N as F * F::EPSILON;
    for i in (0..N).rev() {
        let mut curr = b[i];
        for j in i..N {
            curr -= out[j] * u[i][j];
        }
        // explicitly skip values which are near 0.
        // FIXME, decide whether this makes sense if u[i][i] also near 0.
        if curr.abs() <= rcond {
            continue;
        }
        out[i] = curr / u[i][i];
    }
    out
}

/// QR solve a fixed size matrix.
// TODO need to handle case where R < C (Underdetermined matrix)
pub fn qr_solve<const R: usize, const C: usize>(
    q: Matrix<R, C>,
    r: Matrix<C, C>,
    b: [F; R],
) -> [F; C] {
    assert!(
        R >= C,
        "Underdetermined system, decompose A^T and pass to qr_solve_underdetermined"
    );
    let qtb = vecmul(transpose(q), b);
    upper_right_triangular_solve(r, qtb)
}

fn vecmul<const R: usize, const C: usize>(a: Matrix<R, C>, b: [F; C]) -> [F; R] {
    let mut out = [0.; R];
    for i in 0..R {
        for k in 0..C {
            out[i] += a[i][k] * b[k];
        }
    }
    out
}

#[cfg(test)]
fn matmul<const R: usize, const C: usize, const C2: usize>(
    a: Matrix<R, C>,
    b: Matrix<C, C2>,
) -> Matrix<R, C2> {
    let mut out = [[0.; C2]; R];
    for i in 0..R {
        for j in 0..C2 {
            for k in 0..C {
                out[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    out
}

#[test]
fn test_qr_decomp() {
    let a = [[2., 0.], [0., 1.], [2., 2.]];
    let b = [1.; 3];

    let (q, r) = mgs_qr(a);
    //assert_eq!(a, matmul(q,r));
    //assert_eq!(matmul(transpose(q),q), [[1., 0.], [0., 1.]]);

    println!("{r:?}");

    //let s = upper_right_triangular_solve(r, [1., 1.]);
    //assert_eq!([1.; 2], vecmul(r, s));
    let qtb = vecmul(transpose(q), b);
    upper_right_triangular_solve(r, qtb);

    let x = qr_solve(q, r, b);
    assert!((x[0] - 0.3333).abs() < 1e-4);
    assert!((x[1] - 0.3333).abs() < 1e-4);
}

/// Dynamic QR solving with variable number of rows.
/// Uses `a` as a buffer.
pub fn dyn_mgs_qr<const C: usize>(a: &mut [[F; C]], q: &mut Vec<[F; C]>) -> Matrix<C, C> {
    let nr = a.len();
    q.resize(nr, [0.; C]);
    let mut r = [[0.; C]; C];

    for i in 0..C {
        r[i][i] = norm_iter((0..nr).map(|ri| a[ri][i]));
        if r[i][i] < F::EPSILON {
            continue;
        }
        let irii = r[i][i].recip();
        assert!(irii.is_finite());
        for ri in 0..nr {
            q[ri][i] = irii * a[ri][i];
        }

        for j in i..C {
            r[i][j] = (0..nr).map(|ri| a[ri][j] * q[ri][i]).sum::<F>();
            let rij = r[i][j];
            for ri in 0..nr {
                a[ri][j] -= rij * q[ri][i];
            }
        }
    }

    r
}

/// (RxC)^T (Rx1)
fn dyn_transpose_vecmul<const C: usize>(a: &[[F; C]], b: impl Fn(usize) -> F) -> [F; C] {
    let mut out = [0.; C];
    let nr = a.len();
    for i in 0..nr {
        let b_i = b(i);
        for j in 0..C {
            out[j] += a[i][j] * b_i;
        }
    }
    out
}

/// Solves the linear system QRx = b, where the number of rows in Q & b is dynamic, but the
/// number of columns is fixed.
pub fn dyn_qr_solve<const C: usize>(
    q: &[[F; C]],
    r: Matrix<C, C>,
    b: impl Fn(usize) -> F,
) -> [F; C] {
    assert!(q.len() >= C, "TODO handle case where there are fewer rows and cols");
    upper_right_triangular_solve(r, dyn_transpose_vecmul(q, b))
}

#[test]
fn test_dyn_qr_decomp() {
    let mut v = [[2., 0.], [0., 1.], [2., 2.]];
    let (q, _r) = mgs_qr(v);

    let mut dq = vec![];
    let dr = dyn_mgs_qr(&mut v, &mut dq);

    assert_eq!(
        dyn_transpose_vecmul(&dq, |_| 1.),
        vecmul(transpose(q), [1.; 3])
    );

    let x = dyn_qr_solve(&dq, dr, |_| 1.);
    assert!((x[0] - 0.33333).abs() < 1e-3);
    assert!((x[1] - 0.33333).abs() < 1e-3);
}

/// Solve a system defined by `U^t x = b`
// TODO need to figure out why this is broken?
fn lower_tri_solve<const R: usize, const C: usize>(u: Matrix<C, C>, b: [F; C]) -> [F; R] {
    let mut out = [0.; R];
    for i in 0..C {
        let mut curr = b[i];
        for j in 0..i {
            curr -= out[j] * u[i][j];
        }
        out[i] = curr / u[i][i];
        assert!(out[i].is_finite());
    }
    out
}

/// QR solve an underdetermined system, where R < C.
/// When computing the QR factorization, compute it for A^T instead.
pub fn qr_solve_underdetermined<const R: usize, const C: usize>(
    q: Matrix<C, R>,
    r: Matrix<R, R>,
    b: [F; R],
) -> [F; C] {
    vecmul(q, lower_tri_solve(r, b))
}

#[test]
fn test_qr_underdetermined() {
    let a: Matrix<3, 4> = [[1., 1., 1., 1.], [-1., 1., -1., 1.], [1., 1., -1., 1.]];
    let (q, r): (Matrix<4, 3>, Matrix<3, 3>) = mgs_qr(transpose(a));
    let b = [1.; 3];
    let x: [F; 4] = qr_solve_underdetermined(q, transpose(r), b);
    assert_eq!(vecmul(a, x), b, "{x:?}");
}

