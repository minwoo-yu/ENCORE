#include <cstring>  // for memcmp
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "cuda_utils.h"

CudaArrayManaged::CudaArrayManaged() : cuArray(nullptr), texObj(0), cached_W(-1), cached_H(-1), cached_L(-1) {
    std::memset(&cached_texDesc, 0, sizeof(cudaTextureDesc));
}
CudaArrayManaged::~CudaArrayManaged() {
    if (texObj) cudaDestroyTextureObject(texObj);
    if (cuArray) cudaFreeArray(cuArray);
}

bool CudaArrayManaged::update(int width, int height, int layers) {
    if (cuArray == nullptr || cached_W != width || cached_H != height || cached_L < layers) {
        if (texObj) {
            cudaDestroyTextureObject(texObj);
            texObj = 0;
        }
        if (cuArray) cudaFreeArray(cuArray);

        cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
        cudaExtent extent = make_cudaExtent(width, height, layers);
        cudaError_t err = cudaMalloc3DArray(&cuArray, &desc, extent, cudaArrayLayered);
        if (err != cudaSuccess) {
            printf("CUDA error in cudaMalloc3DArray: %s (W=%d, H=%d, L=%d)\n", cudaGetErrorString(err), width, height,
                   layers);
            return false;
        }
        cached_W = width;
        cached_H = height;
        cached_L = layers;
    }
    return true;
}

void CudaArrayManaged::copyFrom(const void* src, size_t srcPitch, int width, int height, int layers,
                                cudaStream_t stream) {
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr((void*)src, srcPitch, width, height);
    copyParams.dstArray = cuArray;
    copyParams.extent = make_cudaExtent(width, height, layers);
    copyParams.kind = cudaMemcpyDeviceToDevice;
    cudaMemcpy3DAsync(&copyParams, stream);
}

cudaTextureObject_t CudaArrayManaged::getTexture(const cudaTextureDesc& texDesc) {
    // If the texture object already exists and the description hasn't changed, reuse it.
    if (texObj != 0 && std::memcmp(&texDesc, &cached_texDesc, sizeof(cudaTextureDesc)) == 0) {
        return texObj;
    }

    if (texObj != 0) {
        cudaDestroyTextureObject(texObj);
        texObj = 0;
    }

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    cached_texDesc = texDesc;
    return texObj;
}

// ----------------------------------------------------------------------------
// TexturePool Implementation
// ----------------------------------------------------------------------------

struct PoolKey {
    int dev, w, h, l, tag;
    bool operator==(const PoolKey& o) const { return dev == o.dev && w == o.w && h == o.h && l == o.l && tag == o.tag; }
};

struct PoolKeyHash {
    size_t operator()(const PoolKey& k) const {
        size_t h = k.dev;
        h ^= (size_t)k.w << 8;
        h ^= (size_t)k.h << 16;
        h ^= (size_t)k.l << 24;    // num layers
        h ^= (size_t)k.tag << 31;  // 0: for weight, 1: for bias
        return h;
    }
};

static std::unordered_map<PoolKey, std::list<PoolKey>::iterator, PoolKeyHash> pool_map;
static std::unordered_map<PoolKey, std::unique_ptr<CudaArrayManaged>, PoolKeyHash> pool_data;
static std::list<PoolKey> pool_lru;
static std::mutex pool_mutex;
static const int MAX_POOL_SIZE = 64;  // Approx 32 cases * 2 tags

CudaArrayManaged& TexturePool::get(int w, int h, int l, int tag) {
    std::lock_guard<std::mutex> lock(pool_mutex);
    int dev;
    cudaGetDevice(&dev);
    PoolKey key = {dev, w, h, l, tag};

    auto it = pool_map.find(key);
    if (it != pool_map.end()) {
        // Move to front (most recently used)
        pool_lru.erase(it->second);
        pool_lru.push_front(key);
        it->second = pool_lru.begin();
        return *pool_data[key].get();
    }

    // Handle eviction
    if (pool_lru.size() >= MAX_POOL_SIZE) {
        PoolKey old_key = pool_lru.back();
        pool_lru.pop_back();
        pool_map.erase(old_key);
        pool_data.erase(old_key);
    }

    // Create new
    auto managed = std::make_unique<CudaArrayManaged>();
    managed->update(w, h, l);

    pool_lru.push_front(key);
    pool_map[key] = pool_lru.begin();
    pool_data[key] = std::move(managed);

    return *pool_data[key].get();
}

void TexturePool::clear() {
    std::lock_guard<std::mutex> lock(pool_mutex);
    pool_data.clear();
    pool_map.clear();
    pool_lru.clear();
}
