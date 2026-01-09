//! SIFT (Scale-Invariant Feature Transform) implementation in Rust
//!
//! This library provides functionality to detect and describe keypoints in images
//! using the SIFT algorithm.

pub mod detector;
pub mod descriptor;
pub mod keypoint;
pub mod gaussian_blur;

pub use detector::SiftDetector;
pub use descriptor::SiftDescriptor;
pub use keypoint::KeyPoint;