use std::sync::Arc;

use vulkano::{
    app_info_from_cargo_toml,
    buffer::{BufferUsage, ImmutableBuffer},
    command_buffer::{AutoCommandBufferBuilder, DynamicState},
    device::{Device, DeviceExtensions, Features},
    framebuffer::{Framebuffer, Subpass},
    image::ImageUsage,
    impl_vertex,
    instance::{
        debug::{DebugCallback, DebugCallbackCreationError},
        layers_list, Instance, InstanceExtensions, PhysicalDevice, QueueFamily,
    },
    pipeline::{viewport::Viewport, GraphicsPipeline},
    single_pass_renderpass,
    swapchain::{
        self, AcquireError, ColorSpace, CompositeAlpha, PresentMode, SurfaceTransform, Swapchain,
        SwapchainCreationError,
    },
    sync::{FlushError, GpuFuture},
};
use vulkano_win::VkSurfaceBuild;
use winit::{EventsLoop, WindowBuilder};

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
#version 450

layout(location = 0) in vec2 position;

void main() {
    gl_Position = vec4(position, 0, 1);
}"
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
#version 450

layout(location = 0) out vec4 f_color;

void main() {
    f_color = vec4(1, 0, 0, 1);
}"
    }
}

#[derive(Default, Copy, Clone)]
struct Vertex {
    position: [f32; 2],
}

impl_vertex!(Vertex, position);

fn main() {
    println!("Supported layers:");
    for layer in layers_list().unwrap() {
        println!("\t{}", layer.name());
    }

    let extensions = InstanceExtensions {
        ext_debug_utils: true,
        ..vulkano_win::required_extensions()
    };
    let layers = vec!["VK_LAYER_LUNARG_standard_validation"];
    let instance = Instance::new(Some(&app_info_from_cargo_toml!()), &extensions, layers).unwrap();

    let _debug_callback = setup_debug_callback(&instance).unwrap();

    let physical_device = find_physical_device(&instance).unwrap();
    let queue_family = physical_device
        .queue_families()
        .find(QueueFamily::supports_graphics)
        .unwrap();

    println!(
        "Supported device extensions: {:#?}",
        DeviceExtensions::supported_by_device(physical_device)
    );

    let features = Features::none();
    let extensions = DeviceExtensions {
        khr_storage_buffer_storage_class: true,
        khr_swapchain: true,
        ..DeviceExtensions::none()
    };
    let queue_families = vec![(queue_family, 0.5)];
    let (device, mut queues) =
        Device::new(physical_device, &features, &extensions, queue_families).unwrap();
    let queue = queues.next().unwrap();

    let mut events_loop = EventsLoop::new();
    let surface = WindowBuilder::new()
        .with_title("Water")
        .build_vk_surface(&events_loop, instance.clone())
        .unwrap();

    let caps = surface.capabilities(physical_device).unwrap();
    println!("caps = {:#?}", caps);

    let mut dimensions = caps.current_extent.unwrap_or([1280, 720]);
    let alpha = if caps.supported_composite_alpha.opaque {
        CompositeAlpha::Opaque
    } else {
        caps.supported_composite_alpha.iter().next().unwrap()
    };
    let (format, color_space) = caps
        .supported_formats
        .iter()
        .copied()
        .find(|(_, cs)| *cs == ColorSpace::SrgbNonLinear)
        .unwrap_or(caps.supported_formats[0]);
    let image_count = if let Some(max) = caps.max_image_count {
        caps.min_image_count.max(max.min(2))
    } else {
        caps.min_image_count.max(2)
    };

    let (mut swapchain, mut images) = Swapchain::new(
        device.clone(),
        surface.clone(),
        image_count,
        format,
        dimensions,
        1,
        ImageUsage {
            color_attachment: true,
            ..ImageUsage::none()
        },
        &queue,
        SurfaceTransform::Identity,
        alpha,
        PresentMode::Fifo,
        true,
        color_space,
    )
    .unwrap();

    let vertices = [
        Vertex {
            position: [-0.5, -0.5],
        },
        Vertex {
            position: [0., 0.5],
        },
        Vertex {
            position: [0.5, -0.25],
        },
    ];
    let (vertex_buffer, buffer_future) = ImmutableBuffer::from_iter(
        vertices.iter().copied(),
        BufferUsage::vertex_buffer(),
        queue.clone(),
    )
    .unwrap();

    buffer_future
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    let render_pass = Arc::new(
        single_pass_renderpass!(device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: format,
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {}
            }
        )
        .unwrap(),
    );

    let vs = vs::Shader::load(device.clone()).unwrap();
    let fs = fs::Shader::load(device.clone()).unwrap();

    let pipeline = Arc::new(
        GraphicsPipeline::start()
            .vertex_input_single_buffer::<Vertex>()
            .vertex_shader(vs.main_entry_point(), ())
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(fs.main_entry_point(), ())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap(),
    );

    let mut recreate_swapchain = false;
    loop {
        let mut quit = false;
        events_loop.poll_events(|event| match event {
            winit::Event::WindowEvent {
                event: winit::WindowEvent::CloseRequested,
                ..
            } => quit = true,
            winit::Event::WindowEvent {
                event: winit::WindowEvent::Resized(logical_size),
                ..
            } => {
                // If we don't handle this, on Wayland the buffer will stay the same size.
                let new_dimensions: (u32, u32) = logical_size.into();
                dimensions = [new_dimensions.0, new_dimensions.1];
                recreate_swapchain = true;
            }
            _ => (),
        });

        if quit {
            break;
        }

        if recreate_swapchain {
            let caps = surface.capabilities(physical_device).unwrap();
            dimensions = caps.current_extent.unwrap_or(dimensions);
            let new_swapchain = match swapchain.recreate_with_dimension(dimensions) {
                Ok(x) => x,
                Err(SwapchainCreationError::UnsupportedDimensions) => {
                    // println!("Swapchain race condition?");
                    continue;
                }
                Err(err) => panic!("{:?}", err),
            };
            recreate_swapchain = false;
            swapchain = new_swapchain.0;
            images = new_swapchain.1;
            continue;
        }

        let (image_num, acquire_future) =
            match swapchain::acquire_next_image(swapchain.clone(), None) {
                Ok(x) => x,
                Err(AcquireError::OutOfDate) => {
                    // This happens on X11.
                    recreate_swapchain = true;
                    continue;
                }
                Err(err) => panic!("{:?}", err),
            };

        let framebuffer = Arc::new(
            Framebuffer::start(render_pass.clone())
                .add(images[image_num].clone())
                .unwrap()
                .build()
                .unwrap(),
        );

        let dynamic_state = DynamicState {
            viewports: Some(vec![Viewport {
                origin: [0., 0.],
                dimensions: [dimensions[0] as f32, dimensions[1] as f32],
                depth_range: 0.0..1.,
            }]),
            ..DynamicState::none()
        };

        let command_buffer =
            AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue_family)
                .unwrap()
                .begin_render_pass(framebuffer.clone(), false, vec![[0., 0., 0., 1.].into()])
                .unwrap()
                .draw(
                    pipeline.clone(),
                    &dynamic_state,
                    vertex_buffer.clone(),
                    (),
                    (),
                )
                .unwrap()
                .end_render_pass()
                .unwrap()
                .build()
                .unwrap();

        // X11 can return OutOfDate on both of these.
        let fence = match acquire_future
            .then_execute(queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
            .then_signal_fence_and_flush()
        {
            Ok(x) => x,
            Err(FlushError::OutOfDate) => continue,
            Err(err) => panic!("{:?}", err),
        };
        match fence.wait(None) {
            Ok(()) => (),
            Err(FlushError::OutOfDate) => continue,
            Err(err) => panic!("{:?}", err),
        }
    }
}

fn setup_debug_callback(
    instance: &Arc<Instance>,
) -> Result<DebugCallback, DebugCallbackCreationError> {
    DebugCallback::errors_and_warnings(instance, |msg| {
        let severity = if msg.severity.error {
            "error"
        } else if msg.severity.warning {
            "warning"
        } else if msg.severity.information {
            "information"
        } else if msg.severity.verbose {
            "verbose"
        } else {
            panic!("no-impl");
        };

        let ty = if msg.ty.general {
            "general"
        } else if msg.ty.validation {
            "validation"
        } else if msg.ty.performance {
            "performance"
        } else {
            panic!("no-impl");
        };

        println!(
            "{} {} {}: {}",
            msg.layer_prefix, ty, severity, msg.description
        );
    })
}

fn find_physical_device(instance: &Arc<Instance>) -> Option<PhysicalDevice> {
    println!("Physical devices:");
    for device in PhysicalDevice::enumerate(instance) {
        println!(
            "\t{}: {} ({:?})",
            device.index(),
            device.name(),
            device.ty()
        );
    }

    PhysicalDevice::from_index(instance, 0)
}
