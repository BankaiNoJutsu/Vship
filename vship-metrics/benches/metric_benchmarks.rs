// Benchmarks for vship-metrics

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use vship_metrics::{MetricsContext, ImageData, ImageFormat, Metric};

fn create_test_image(width: u32, height: u32) -> ImageData {
    let mut img = ImageData::new(width, height, ImageFormat::RGB);

    // Fill with some pattern
    for i in 0..img.data.len() {
        img.data[i] = ((i % 256) as f32) / 255.0;
    }

    img
}

fn benchmark_ssimulacra2(c: &mut Criterion) {
    let mut group = c.benchmark_group("ssimulacra2");

    let sizes = vec![
        (640, 480, "VGA"),
        (1280, 720, "HD"),
        (1920, 1080, "FHD"),
    ];

    for (width, height, name) in sizes {
        let ctx = MetricsContext::new().expect("Failed to create context");
        let mut metric = ctx.create_ssimulacra2().expect("Failed to create metric");

        let reference = create_test_image(width, height);
        let distorted = create_test_image(width, height);

        group.bench_with_input(
            BenchmarkId::new("compute", name),
            &(width, height),
            |b, _| {
                b.iter(|| {
                    metric.compute(black_box(&reference), black_box(&distorted))
                        .expect("Failed to compute");
                });
            },
        );
    }

    group.finish();
}

fn benchmark_butteraugli(c: &mut Criterion) {
    let mut group = c.benchmark_group("butteraugli");

    let sizes = vec![
        (640, 480, "VGA"),
        (1280, 720, "HD"),
        (1920, 1080, "FHD"),
    ];

    for (width, height, name) in sizes {
        let ctx = MetricsContext::new().expect("Failed to create context");
        let mut metric = ctx.create_butteraugli().expect("Failed to create metric");

        let reference = create_test_image(width, height);
        let distorted = create_test_image(width, height);

        group.bench_with_input(
            BenchmarkId::new("compute", name),
            &(width, height),
            |b, _| {
                b.iter(|| {
                    metric.compute(black_box(&reference), black_box(&distorted))
                        .expect("Failed to compute");
                });
            },
        );
    }

    group.finish();
}

fn benchmark_cvvdp(c: &mut Criterion) {
    let mut group = c.benchmark_group("cvvdp");

    let sizes = vec![
        (640, 480, "VGA"),
        (1280, 720, "HD"),
        (1920, 1080, "FHD"),
    ];

    for (width, height, name) in sizes {
        let ctx = MetricsContext::new().expect("Failed to create context");
        let mut metric = ctx.create_cvvdp().expect("Failed to create metric");

        let reference = create_test_image(width, height);
        let distorted = create_test_image(width, height);

        group.bench_with_input(
            BenchmarkId::new("compute", name),
            &(width, height),
            |b, _| {
                b.iter(|| {
                    metric.compute(black_box(&reference), black_box(&distorted))
                        .expect("Failed to compute");
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark_ssimulacra2, benchmark_butteraugli, benchmark_cvvdp);
criterion_main!(benches);
