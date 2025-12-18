use image;
use sift_rs::{SiftDetector, SiftDescriptor};

fn main() {
    // 加载图像并转换为灰度图像
    let img = image::open("./examples/Lenna.jpg").unwrap().to_luma8();
    let mut detector = SiftDetector::new();
    let mut keypoints = detector.detect(img);
    let descriptor = SiftDescriptor::new();
    descriptor.compute_descriptors(&mut keypoints, detector.scale_spaces);
    for keypoint in keypoints.iter_mut() {
        println!("Keypoint at {}, {}, {}, {}, {}, {:?}", keypoint.x, keypoint.y, keypoint.scale, keypoint.octave, keypoint.orientation, keypoint.descriptor);
    }
}