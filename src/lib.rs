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
    for i in (0..N).rev() {
        let mut curr = b[i];
        for j in i..N {
            curr -= out[j] * u[i][j];
        }
        out[i] = curr / u[i][i];
    }
    out
}

pub fn qr_solve<const R: usize, const C: usize>(
    q: Matrix<R, C>,
    r: Matrix<C, C>,
    b: [F; R],
) -> [F; C] {
    let qtb = matmul(transpose(q), transpose([b]));
    upper_right_triangular_solve(r, transpose(qtb)[0])
}

pub fn matmul<const R: usize, const C: usize, const C2: usize>(
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
    let v = [[2., 0.], [0., 1.], [2., 2.]];
    //let v = [[1., 2.], [3., 4.], [5., 6.]];
    let (q, r) = mgs_qr(v);
    let s = upper_right_triangular_solve(r, [1., 1.]);
    assert_eq!([[1.]; 2], matmul(r, transpose([s])));

    let v = qr_solve(q, r, [1., 1., 1.]);

}
