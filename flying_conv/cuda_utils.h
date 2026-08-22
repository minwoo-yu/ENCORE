#pragma once
#include <cuda_runtime.h>

/**
 * @brief Manages a cudaArrayLayered with spatial Dimension and layer count caching.
 * Reallocates only if the spatial dimensions change or the required layers exceed the current allocation.
 */
class CudaArrayManaged {
   public:
    CudaArrayManaged();
    ~CudaArrayManaged();

    // Reallocate if width/height change, or if we need more layers than currently allocated.
    bool update(int width, int height, int layers);

    // Asynchronously copy 3D volume or 2D layered data into the array.
    void copyFrom(const void* src, size_t srcPitch, int width, int height, int layers, cudaStream_t stream);

    // Get the texture object bound to this array, creating one if it doesn't exist.
    cudaTextureObject_t getTexture(const cudaTextureDesc& texDesc);

    cudaArray_t getArray() const { return cuArray; }

   private:
    cudaArray_t cuArray;
    cudaTextureObject_t texObj;
    int cached_W, cached_H, cached_L;
    cudaTextureDesc cached_texDesc;
};

/**
 * @brief Global pool for CudaArrayManaged objects, identifying them by spatial dims and tags.
 */
class TexturePool {
   public:
    static CudaArrayManaged& get(int w, int h, int l, int tag);
    static void clear();  // Useful for memory cleanup if needed
};
