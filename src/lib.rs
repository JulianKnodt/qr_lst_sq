#[cfg(not(feature = "f64"))]
pub type F = f32;

#[cfg(feature = "f64")]
pub type F = f64;

use std::array::from_fn;

type Matrix<const R: usize, const C: usize> = [[F; C]; R];

#[inline]
pub fn normalize<const N: usize>(v: [F; N]) -> [F; N] {
    let sum: F = v.iter().map(|v| v * v).sum();
    if sum < 1e-8 {
        return [0.; N];
    }
    let s = sum.sqrt();
    v.map(|v| v / s)
}

pub fn dot<const N: usize>(a: [F; N], b: [F; N]) -> F {
    (0..N).map(|i| a[i] * b[i]).sum::<F>()
}

pub fn norm<const N: usize>(v: [F; N]) -> F {
    dot(v, v).max(0.).sqrt()
}

pub fn norm_iter(v: impl IntoIterator<Item = F>) -> F {
    v.into_iter().map(|v| v * v).sum::<F>().max(0.).sqrt()
}

pub fn sub<const N: usize>(a: [F; N], b: [F; N]) -> [F; N] {
    from_fn(|i| a[i] - b[i])
}
pub fn kmul<const N: usize>(k: F, a: [F; N]) -> [F; N] {
    a.map(|v| v * k)
}

pub fn transpose<const M: usize, const N: usize>(a: Matrix<M, N>) -> Matrix<N, M> {
    from_fn(|i| from_fn(|j| a[j][i]))
}

pub fn qr<const R: usize, const C: usize>(a: Matrix<R, C>) -> (Matrix<R, C>, Matrix<C, C>) {
    let mut q = [[0.; C]; R];
    let mut r = [[0.; C]; C];

    for j in 0..C {
        // current column
        let mut curr: [_; R] = from_fn(|i| a[i][j]);
        for i in 0..j {
            r[i][j] = dot(q[i], a[j]);
            let q_i = std::array::from_fn(|j| q[j][i]);
            curr = sub(curr, kmul(r[i][j], q_i));
        }

        let rjj = norm(curr);
        r[j][j] = rjj;
        let irjj = rjj.recip();

        for i in 0..R {
            q[i][j] = irjj * curr[i];
        }
    }
    (q, r)
}

pub fn mgs_qr<const R: usize, const C: usize>(a: Matrix<R, C>) -> (Matrix<R, C>, Matrix<C, C>) {
    let mut q = [[0.; C]; R];
    let mut r = [[0.; C]; C];

    // transpose for later convenience
    let mut vs: Matrix<C, R> = from_fn(|i| from_fn(|j| a[j][i]));

    for i in 0..C {
        let v: [_; R] = vs[i];
        r[i][i] = norm(v);
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

pub fn upper_right_triangular_solve<const N: usize>(u: Matrix<N, N>, b: [F; N]) -> [F; N] {
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

/// Solve a system defined by `U^t x = b`
fn right_tri_transposed_solve<const R: usize, const C: usize>(
    u: Matrix<R, R>,
    b: [F; C],
) -> [F; C] {
    assert!(C < R);
    let mut out = [0.; C];
    let rcond = R as F * F::EPSILON;
    for i in 0..C {
        let mut curr = b[i];
        for j in 0..i {
            curr -= out[j] * u[j][i];
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

/// QR solve an underdetermined system, where R < C.
/// When computing the QR factorization, compute it for A^T instead.
pub fn qr_solve_underdetermined<const R: usize, const C: usize>(
    q: Matrix<C, R>,
    r: Matrix<C, C>,
    b: [F; R],
) -> [F; C] {
    assert!(R < C);
    let rb = right_tri_transposed_solve(r, b);
    vecmul(q, rb)
}

pub fn vecmul<const R: usize, const C: usize>(a: Matrix<R, C>, b: [F; C]) -> [F; R] {
    let mut out = [0.; R];
    for i in 0..R {
        for k in 0..C {
            out[i] += a[i][k] * b[k];
        }
    }
    out
}

/*
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
*/

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
        let irii = r[i][i].recip();
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
pub fn dyn_transpose_vecmul<const C: usize>(a: &[[F; C]], b: impl Fn(usize) -> F) -> [F; C] {
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

pub fn dyn_qr_solve<const C: usize>(
    q: &[[F; C]],
    r: Matrix<C, C>,
    b: impl Fn(usize) -> F,
) -> [F; C] {
    assert!(q.len() >= C);
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
    println!("{x:?}");
    assert!((x[0] - 0.33333).abs() < 1e-3);
    assert!((x[1] - 0.33333).abs() < 1e-3);
}
