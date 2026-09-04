//! Hardware abstraction layer for Intel Gaudi2 (HL-225) accelerators.
//!
//! Device discovery is done through the kernel `accel` subsystem exposed under
//! `/sys/class/accel`. Reading sysfs works with both the in-tree and the
//! out-of-tree `habanalabs` drivers and does not require the SynapseAI
//! userspace to be installed, which makes it usable for inventory, stepping
//! detection, and health checks before any compute stack is brought up.

use std::fs;
use std::path::Path;

use reng_core::{DeviceId, DeviceKind, Result};

/// Root of the kernel `accel` class.
const ACCEL_CLASS: &str = "/sys/class/accel";

/// Silicon stepping of a Gaudi2 die (for example `A0`), parsed from its fuse
/// version string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stepping(pub String);

/// A discovered Gaudi2 accelerator.
#[derive(Debug, Clone)]
pub struct GaudiDevice {
    /// Index of the device (the `N` in `accelN`).
    pub id: DeviceId,
    /// Kind of device.
    pub kind: DeviceKind,
    /// PCI address (for example `0000:cc:00.0`) if it could be resolved.
    pub pci_addr: Option<String>,
    /// Silicon stepping if it could be parsed from `fuse_ver`.
    pub stepping: Option<Stepping>,
}

/// Parse the silicon stepping out of a Gaudi2 `fuse_ver` string such as
/// `01P0-HL2080A0-15-TNPS09-13-04-06`.
///
/// The token of interest is `HL2080<stepping>`, where `<stepping>` is a letter
/// followed by a digit (for example `A0`). Returns `None` if no such token is
/// present.
#[must_use]
pub fn parse_stepping(fuse_ver: &str) -> Option<Stepping> {
    for token in fuse_ver.split('-') {
        if let Some(rest) = token.strip_prefix("HL2080") {
            let bytes = rest.as_bytes();
            if bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1].is_ascii_digit() {
                return Some(Stepping(rest[..2].to_string()));
            }
        }
    }
    None
}

/// Enumerate Gaudi2 devices visible to the host.
///
/// Returns an empty vector (not an error) on hosts without the `accel` class,
/// for example CI runners that do not have the hardware.
///
/// # Errors
///
/// Returns an error if the `accel` class exists but cannot be read.
pub fn enumerate_devices() -> Result<Vec<GaudiDevice>> {
    enumerate_devices_in(Path::new(ACCEL_CLASS))
}

fn enumerate_devices_in(root: &Path) -> Result<Vec<GaudiDevice>> {
    let mut devices = Vec::new();
    let entries = match fs::read_dir(root) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(devices),
        Err(e) => return Err(e.into()),
    };
    for entry in entries {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        // Match `accelN` device nodes, skipping the `accel_controlN` control
        // nodes and anything that is not a plain numeric index.
        let Some(idx) = name.strip_prefix("accel") else {
            continue;
        };
        if idx.starts_with('_') {
            continue;
        }
        let Ok(id) = idx.parse::<u32>() else {
            continue;
        };

        let dev_dir = entry.path().join("device");
        let pci_addr = fs::read_link(&dev_dir)
            .ok()
            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()));
        let stepping = read_trimmed(&dev_dir.join("fuse_ver")).and_then(|s| parse_stepping(&s));

        devices.push(GaudiDevice {
            id: DeviceId(id),
            kind: DeviceKind::Gaudi2,
            pci_addr,
            stepping,
        });
    }
    devices.sort_by_key(|d| d.id.0);
    Ok(devices)
}

/// Read a sysfs attribute and trim surrounding whitespace, returning `None`
/// when the file is missing or empty.
fn read_trimmed(path: &Path) -> Option<String> {
    fs::read_to_string(path)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;

    #[test]
    fn parses_a0_stepping() {
        let s = parse_stepping("01P0-HL2080A0-15-TNPS09-13-04-06");
        assert_eq!(s, Some(Stepping("A0".to_string())));
    }

    #[test]
    fn parses_hypothetical_later_stepping() {
        assert_eq!(
            parse_stepping("01P0-HL2080B1-15-TNPS09"),
            Some(Stepping("B1".to_string()))
        );
    }

    #[test]
    fn rejects_unrelated_strings() {
        assert_eq!(parse_stepping("no-stepping-here"), None);
        assert_eq!(parse_stepping(""), None);
        assert_eq!(parse_stepping("HL2080"), None);
    }

    #[test]
    fn enumerate_missing_class_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("does-not-exist");
        assert!(enumerate_devices_in(&missing).unwrap().is_empty());
    }

    #[test]
    fn enumerate_reads_fake_sysfs() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();

        // accel0 -> device symlink to a fake PCI dir carrying fuse_ver.
        let pci = root.join("pci").join("0000:cc:00.0");
        fs::create_dir_all(&pci).unwrap();
        fs::write(pci.join("fuse_ver"), "01P0-HL2080A0-15-TNPS09-13-04-06\n").unwrap();
        fs::create_dir_all(root.join("accel0")).unwrap();
        symlink(&pci, root.join("accel0").join("device")).unwrap();

        // A control node that must be ignored.
        fs::create_dir_all(root.join("accel_controls0")).unwrap();

        let devs = enumerate_devices_in(root).unwrap();
        assert_eq!(devs.len(), 1);
        assert_eq!(devs[0].id, DeviceId(0));
        assert_eq!(devs[0].kind, DeviceKind::Gaudi2);
        assert_eq!(devs[0].pci_addr.as_deref(), Some("0000:cc:00.0"));
        assert_eq!(devs[0].stepping, Some(Stepping("A0".to_string())));
    }
}
