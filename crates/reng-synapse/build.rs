//! Link libSynapse only when the `link-synapse` feature is enabled, so the
//! crate still builds as a plain rlib on hosts without the Habana libraries.

fn main() {
    if std::env::var_os("CARGO_FEATURE_LINK_SYNAPSE").is_some() {
        let dir =
            std::env::var("SYNAPSE_LIB_DIR").unwrap_or_else(|_| "/usr/lib/habanalabs".to_string());
        println!("cargo:rustc-link-search=native={dir}");
        println!("cargo:rustc-link-lib=dylib=Synapse");
    }
}
