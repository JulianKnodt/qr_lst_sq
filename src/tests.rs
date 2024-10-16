use quickcheck::quickcheck;

use super::{matmul, mgs_qr, F};

quickcheck! {
    fn qr_works(a: [[F; 4]; 4]) -> bool {
        let (q,r) = mgs_qr(a);
        let new_a = matmul(q,r);
        (0..4).all(|i| {
          (0..4).all(|j| {
            (new_a[i][j] - a[i][j]).abs() < 1e-5
          })
        })
    }
}
