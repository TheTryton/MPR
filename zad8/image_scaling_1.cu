#include <stdio.h>
#include <scrImagePgmPpmPackage.hpp>
#include <common.hpp>

//Kernel which calculate the resized image
__global__ void createResizedImage(std::byte* imageScaledData, int scaled_width, float scale_factor, cudaTextureObject_t texObj)
{
	const unsigned int tidX = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int tidY = blockIdx.y*blockDim.y + threadIdx.y;
	const unsigned index = tidY*scaled_width+tidX;
       	
	// Step 4: Read the texture memory from your texture reference in CUDA Kernel
	imageScaledData[index] = static_cast<std::byte>(tex2D<unsigned char>(texObj,(float)(tidX*scale_factor),(float)(tidY*scale_factor)));
}

constexpr std::array<std::pair<std::string_view, std::string_view>, 2> images
{
	std::pair{"aerosmith-double.pgm", "aerosmith-double-scaled.pgm"},
	std::pair{"voyager2.pgm", "voyager2-scaled.pgm"},
};

constexpr std::array<float, 8> scaling_ratios
{
	0.25f,
	0.5f,
	0.75f,
	1.0f,
	1.5f,
	2.0f,
	3.0f,
	4.0f,
};

void print_header()
{
	using std::cout;
	using std::endl;
	using std::setw;
	using std::left;

	cout <<
		setw(12) << left << "width" << ", " <<
		setw(12) << left << "height" << ", " <<
		setw(12) << left << "scaling ratio" << ", " <<
		setw(12) << left << "scaled width" << ", " <<
		setw(12) << left << "scaled height" << ", " <<
		setw(12) << left << "Block Size" << ", " <<
		setw(12) << left << "GPU time [s]" <<
		endl;
}


int main(int argc, char*argv[])
{
	using std::cout;
	using std::endl;
	using std::setw;
	using std::left;
	using std::chrono::duration_cast;

	try
	{
		auto device_count = get_devices_count();
		for (int32_t i = 0; i < device_count; i++)
		{
			auto p = get_device_properties(i);

			cout << "Device with ID=" << i << endl;
			print_properties(p);
		}

		auto current_device_i = get_current_device();
		auto current_device_p = get_device_properties(current_device_i);

		cout << endl;
		cout << "Currently selected device ID=" << current_device_i << endl;
		cout << endl;

		print_header();

		for (auto scaling_ratio : scaling_ratios)
		{
			for (size_t block_size = 2; block_size < 64; block_size *= 2)
			{
				for (auto& [input, output] : images)
				{
					int width = 0;
					int height = 0;
					int scaled_height = 0;
					int scaled_width = 0;
					seconds_double time;

					{
						get_PgmPpmParams(input.data(), &height, &width);
						auto image_data = allocate_host_memory<std::byte>(width * height);
						scr_read_pgm(input.data(), reinterpret_cast<unsigned char*>(image_data.get()), height, width);

						auto start = std::chrono::high_resolution_clock::now();

						scaled_height = static_cast<int>(height * scaling_ratio);
						scaled_width = static_cast<int>(width * scaling_ratio);
						auto scaled_image_data = allocate_host_memory<std::byte>(scaled_width * scaled_height);
						auto scaled_image_data_d = allocate_device_memory<std::byte>(scaled_width * scaled_height);

						auto image_data_d = allocate_device_array(width, height, cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned));
						throw_if_failed(cudaMemcpyToArray(image_data_d.get(), 0, 0, image_data.get(), height * width * sizeof(std::byte), cudaMemcpyHostToDevice));

						cudaResourceDesc resource_desc{};
						resource_desc.resType = cudaResourceTypeArray;
						resource_desc.res.array.array = image_data_d.get();
						cudaTextureDesc texture_desc{};
						texture_desc.addressMode[0] = cudaAddressModeClamp;
						texture_desc.addressMode[1] = cudaAddressModeClamp;
						texture_desc.filterMode = cudaFilterModePoint;
						texture_desc.readMode = cudaReadModeElementType;
						texture_desc.normalizedCoords = 0;

						cudaTextureObject_t texture_object = 0;
						throw_if_failed(cudaCreateTextureObject(&texture_object, &resource_desc, &texture_desc, NULL));

						dim3 dimBlock(static_cast<unsigned int>(block_size), static_cast<unsigned int>(block_size), 1);
						dim3 dimGrid(scaled_width / dimBlock.x, scaled_height / dimBlock.y, 1);

						createResizedImage << <dimGrid, dimBlock >> > (scaled_image_data_d.get(), scaled_width, 1 / scaling_ratio, texture_object);
						throw_if_failed(cudaGetLastError());
						throw_if_failed(cudaDeviceSynchronize());

						throw_if_failed(cudaMemcpy(scaled_image_data.get(), scaled_image_data_d.get(), scaled_height * scaled_width * sizeof(std::byte), cudaMemcpyDeviceToHost));

						throw_if_failed(cudaDestroyTextureObject(texture_object));

						auto end = std::chrono::high_resolution_clock::now();

						time = duration_cast<seconds_double>(end - start);

						scr_write_pgm(output.data(), reinterpret_cast<unsigned char*>(scaled_image_data.get()), scaled_height, scaled_width, "####");
					}

					cout <<
						setw(12) << left << width << ", " <<
						setw(12) << left << height << ", " <<
						setw(12) << left << scaling_ratio << ", " <<
						setw(12) << left << scaled_width << ", " <<
						setw(12) << left << scaled_height << ", " <<
						setw(12) << left << block_size << ", " <<
						setw(12) << left << time.count() <<
						endl;
				}
			}
		}
	}
	catch (const std::runtime_error& error)
	{
		cout << "Error: " << error.what() << endl;
	}
	catch (...)
	{
		cout << "Unknown error!" << endl;
	}
	return 0;
}
