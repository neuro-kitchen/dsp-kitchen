//! Unit-test helpers.

/// Defines a `#[test]` that runs `$body(&client)` on every CubeCL runtime of
/// [`ComputeTarget::for_tests`](dspk_core::compute::ComputeTarget::for_tests) (compiled in, limited by
/// `DSPK_TEST_RUNTIMES`)
/// (through [`dspk_core::compute::ComputeTask`]), so tests never name a runtime.
macro_rules! runtime_test {
    ($name:ident, $body:ident) => {
        #[test]
        fn $name() {
            struct Task;
            impl dspk_core::compute::ComputeTask for Task {
                type Output = ();
                fn run(self, client: cubecl::prelude::Client) {
                    $body(&client)
                }
            }
            let targets = dspk_core::compute::ComputeTarget::for_tests();
            assert!(!targets.is_empty(), "no CubeCL runtime to test (compiled in and allowed by DSPK_TEST_RUNTIMES)");
            for target in targets {
                target.run(Task).expect("compiled-in runtime");
            }
        }
    };
}
