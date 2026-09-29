fn main() {
    // Dark std-widgets (buttons, list, spin box) to match the app's dark theme
    let config = slint_build::CompilerConfiguration::new().with_style("fluent-dark".into());
    slint_build::compile_with_config("ui/app.slint", config).expect("Failed to compile Slint UI definitions");
}
