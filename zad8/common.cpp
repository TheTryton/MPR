#include <common.hpp>

void throw_if_failed(const cudaError_t& error)
{
    if (error != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

void device_array_deleter(cudaArray* array)
{
    throw_if_failed(cudaFreeArray(array));
};
std::unique_ptr<cudaArray, device_array_deleter_t> allocate_device_array(size_t width, size_t height, cudaChannelFormatDesc channel_desc)
{
    cudaArray* array;
    throw_if_failed(cudaMallocArray(&array, &channel_desc, width, height));
    return std::unique_ptr<cudaArray, device_array_deleter_t>(array, device_array_deleter);
}

int32_t get_devices_count()
{
    int device_count = 0;
    throw_if_failed(cudaGetDeviceCount(&device_count));
    return device_count;
}

int32_t get_current_device()
{
    int current_device = 0;
    throw_if_failed(cudaGetDevice(&current_device));
    return current_device;
}

properties_t get_device_properties(int32_t device)
{
    cudaDeviceProp properties;
    throw_if_failed(cudaGetDeviceProperties(&properties, device));
    properties_t props;

    std::copy_n(std::begin(properties.name), std::size(props.name.name), std::begin(props.name.name));
    std::copy_n(std::begin(properties.uuid.bytes), std::size(props.uuid.bytes), std::begin(props.uuid.bytes));
    std::copy_n(std::begin(properties.luid), std::size(props.luid.bytes), std::begin(props.luid.bytes));

    props.shared_memory_per_block = properties.sharedMemPerBlock;

    props.registers_per_block = properties.regsPerBlock;
    props.registers_per_multiprocessor = properties.regsPerMultiprocessor;
    props.max_blocks_per_multiprocessor = properties.maxBlocksPerMultiProcessor;
    props.number_of_multiprocessors = properties.multiProcessorCount;

    props.max_threads_per_block = properties.maxThreadsPerBlock;
    props.max_threads_dimensions = { properties.maxThreadsDim[0], properties.maxThreadsDim[1], properties.maxThreadsDim[2], };
    props.max_grid_size = { properties.maxGridSize[0], properties.maxGridSize[1], properties.maxGridSize[2], };
    
    props.major_compute_capabilities = properties.major;
    props.minor_compute_capabilities = properties.minor;

    props.clock_frequency_khz = properties.clockRate;
    props.memory_frequency_khz = properties.memoryClockRate;

    props.total_global_memory = properties.totalGlobalMem;
    props.total_constant_memory = properties.totalConstMem;
    props.shared_memory_per_block = properties.sharedMemPerBlock;
    props.shared_memory_per_multiprocessor = properties.sharedMemPerMultiprocessor;
    props.memory_bus_width = properties.memoryBusWidth;
    props.memory_pitch = properties.memPitch;

    props.max_texture_1d_size = { properties.maxTexture1D, };
    props.max_texture_2d_size = { properties.maxTexture2D[0], properties.maxTexture2D[1], };
    props.max_texture_3d_size = { properties.maxTexture3D[0], properties.maxTexture3D[1], properties.maxTexture3D[2], };

    return props;
}

std::ostream& operator<<(std::ostream& o, const name_t& v)
{
    std::string_view view = std::string_view(std::data(v.name), std::size(v.name));
    o << view;
    return o;
}

std::ostream& operator<<(std::ostream& o, const uuid_t& v)
{
    int32_t d0;
    std::memcpy(static_cast<void*>(&d0), v.bytes, sizeof(d0));
    int16_t d1;
    std::memcpy(static_cast<void*>(&d1), v.bytes + sizeof(d0), sizeof(d1));
    int16_t d2;
    std::memcpy(static_cast<void*>(&d2), v.bytes + sizeof(d0) + sizeof(d1), sizeof(d2));
    int16_t d3;
    std::memcpy(static_cast<void*>(&d3), v.bytes + sizeof(d0) + sizeof(d1) + sizeof(d2), sizeof(d3));
    int8_t d4[6];
    std::memcpy(d4, v.bytes + sizeof(d0) + sizeof(d1) + sizeof(d2) + sizeof(d3), std::size(d4));

    constexpr const char* format = "%08lX-%04hX-%04hX-%04hX-%02hhX%02hhX%02hhX%02hhX%02hhX%02hhX";

    char buffer[37];
    sprintf_s(
        buffer,
        format,
        d0,
        d1,
        d2,
        d3,
        d4[0], d4[1], d4[2], d4[3], d4[4], d4[5]
        );

    std::string_view view = std::string_view(std::data(buffer), std::size(buffer));
    o << view;
    return o;
}

std::ostream& operator<<(std::ostream& o, const luid_t& v)
{
    int32_t d0;
    std::memcpy(static_cast<void*>(&d0), v.bytes, sizeof(d0));
    int32_t d1;
    std::memcpy(static_cast<void*>(&d1), v.bytes + sizeof(d0), sizeof(d1));

    constexpr const char* format = "%08lX-%08lX";

    char buffer[18];
    sprintf_s(
        buffer,
        format,
        d0,
        d1
    );

    std::string_view view = std::string_view(std::data(buffer), std::size(buffer));
    o << view;
    return o;
}

void print_properties(const properties_t& p)
{
    using std::cout;
    using std::endl;

    cout << "\tname = \t\t\t\t" << p.name << endl;
    cout << "\tuuid = \t\t\t\t" << p.uuid << endl;
    cout << "\tluid = \t\t\t\t" << p.luid << endl;

    cout << "\tcompute capabilities = \t\t" <<
        p.major_compute_capabilities << '.' <<
        p.minor_compute_capabilities <<
        endl;

    cout << "\tclock frequency = \t\t" << p.clock_frequency_khz / 1e3 << "MHz" << endl;
    cout << "\tmemory frequency = \t\t" << p.memory_frequency_khz / 1e3 << "MHz" << endl;

    cout << "\ttotal global memory = \t\t" << p.total_global_memory / 1e6 << "MB" << endl;
    cout << "\ttotal constant memory = \t" << p.total_constant_memory / 1e3 << "KB" << endl;
    cout << "\tshared memory per block = \t" << p.shared_memory_per_block / 1e3 << "KB" << endl;
    cout << "\tshm per multiprocessor = \t" << p.shared_memory_per_multiprocessor / 1e3 << "KB" << endl;
    cout << "\tmemory bus width = \t\t" << p.memory_bus_width << "bit" << endl;
    cout << "\tmemory pitch = \t\t\t" << p.memory_pitch << endl;

    cout << "\tnumber of multiprocessors = \t" << p.number_of_multiprocessors << endl;
    cout << "\tregisters per block = \t\t" << p.registers_per_block << endl;
    cout << "\tregisters per multiprocessor = \t" << p.registers_per_multiprocessor << endl;
    cout << "\tmax blocks per multiprocessor = " << p.max_blocks_per_multiprocessor << endl;

    cout << "\tmax threads per block = \t" << p.max_threads_per_block << endl;
    cout << "\tmax threads dimensions = \t[" <<
        p.max_threads_dimensions[0] << ", " <<
        p.max_threads_dimensions[1] << ", " <<
        p.max_threads_dimensions[2] << ']' <<
        endl;
    cout << "\tmax grid size = \t\t[" <<
        p.max_grid_size[0] << ", " <<
        p.max_grid_size[1] << ", " <<
        p.max_grid_size[2] << ']' <<
        endl;

    cout << "\tmax texture1D size = \t\t[" <<
        p.max_texture_1d_size[0] << ']' <<
        endl;
    cout << "\tmax texture2D size = \t\t[" <<
        p.max_texture_2d_size[0] << ", " <<
        p.max_texture_2d_size[1] << ']' <<
        endl;
    cout << "\tmax texture3D size = \t\t[" <<
        p.max_texture_3d_size[0] << ", " <<
        p.max_texture_3d_size[1] << ", " <<
        p.max_texture_3d_size[2] << ']' <<
        endl;
}
