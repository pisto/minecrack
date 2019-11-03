#pragma once

#include <stdexcept>
#include <string>
#include <utility>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/*
 * Cuda errors to exceptions translation, same as assertmpi in utilities.h.
 * Use as
 *      cuda*(...) && assertcu;
 */

#define assertcu assertcu_helper{ __FILE__ ":" + std::to_string(__LINE__) }

struct cuda_error : std::runtime_error {
	cuda_error(const std::string &place, cudaError err) : std::runtime_error(
			"@ " + place + ": " + cudaGetErrorString(err)) {}

protected:
	cuda_error(std::string &&err) : std::runtime_error(std::move(err)) {}
};

struct assertcu_helper {
	std::string place;
};

inline int operator&&(cudaError ret, assertcu_helper &&p) {
	return ret == cudaSuccess ? cudaSuccess : throw cuda_error(p.place, ret);
}

/*
 * Routines to get/set global scope variables in device memory
 */

#include <array>

template<typename T>
T *get_device_address(T &on_device) {
	T *address;
	cudaGetSymbolAddress((void **) &address, (const void *) &on_device) && assertcu;
	return address;
}

template<typename T>
T get_device_object(const T &on_device, cudaStream_t stream = 0) {
	T on_host;
	cudaMemcpyFromSymbolAsync((void *) &on_host, (const void *) &on_device, sizeof(T), 0, cudaMemcpyDeviceToHost,
			stream) && assertcu;
	cudaStreamSynchronize(stream) && assertcu;
	return on_host;
}

template<typename T>
void get_device_object(const T &on_device, T &on_host, cudaStream_t stream = 0) {
	cudaMemcpyFromSymbolAsync((void *) &on_host, (const void *) &on_device, sizeof(T), 0, cudaMemcpyDeviceToHost,
			stream) && assertcu;
}

template<typename T, size_t len>
std::array<T, len> get_device_object(const T (&on_device)[len], cudaStream_t stream = 0) {
	std::array<T, len> on_host;
	cudaMemcpyFromSymbolAsync((void *) &on_host, (const void *) &on_device, sizeof(T[len]), 0, cudaMemcpyDeviceToHost,
			stream) && assertcu;
	cudaStreamSynchronize(stream) && assertcu;
	return on_host;
}

template<typename T>
void set_device_object(const T &on_host, T &on_device, cudaStream_t stream = 0) {
	cudaMemcpyToSymbolAsync((const void *) &on_device, (const void *) &on_host, sizeof(T), 0, cudaMemcpyHostToDevice,
			stream) && assertcu;
}

template<typename T>
void memset_device_object(T &on_device, int value, cudaStream_t stream = 0) {
	void *addr;
	cudaGetSymbolAddress(&addr, (const void *) &on_device) && assertcu;
	cudaMemsetAsync(addr, value, sizeof(T), stream) && assertcu;
}

/*
 * Device or host pinned memory wrapper with RAII.
 */

template<typename T = void, bool host = false>
struct cudalist {
	cudalist() = default;

	cudalist(size_t len) {
		static_assert(!host, "wrong constructor");
		cudaMalloc(&mem, len * sizeof(T)) && assertcu;
	}

	cudalist(size_t len, bool wc) {
		static_assert(host, "wrong constructor");
		cudaMallocHost(&mem, len * sizeof(T), cudaHostAllocMapped | (wc ? cudaHostAllocWriteCombined : 0));
	}

	cudalist(cudalist &&o) {
		mem = o.mem;
		o.mem = 0;
	}

	cudalist &operator=(cudalist &&o) {
		this->~cudalist();
		mem = o.mem;
		o.mem = 0;
		return *this;
	}

	operator T *() const { return mem; }

	operator void *() const { return mem; }

	T &operator[](size_t i) { return mem[i]; }

	T *operator*() { return mem; }

	const T &operator[](size_t i) const { return mem[i]; }

	operator bool() const { return mem; }

	T *devptr() const {
		static_assert(host, "cannot call devptr() on a device buffer");
		T *ret;
		cudaHostGetDevicePointer((void **) &ret, (void *) mem, 0) && assertcu;
		return ret;
	}

	~cudalist() {
		if (!mem) return;
		host ? cudaFreeHost((void *) mem) : cudaFree((void *) mem);
		mem = 0;
	}

private:
	T *mem = 0;
};

template<bool host>
struct cudalist<void, host> {
	cudalist() = default;

	cudalist(size_t len) {
		static_assert(!host, "wrong constructor");
		cudaMalloc(&mem, len) && assertcu;
	}

	cudalist(size_t len, bool wc) {
		static_assert(host, "wrong constructor");
		cudaMallocHost(&mem, len, cudaHostAllocMapped | (wc ? cudaHostAllocWriteCombined : 0));
	}

	cudalist(cudalist &&o) {
		mem = o.mem;
		o.mem = 0;
	}

	cudalist &operator=(cudalist &&o) {
		this->~cudalist();
		mem = o.mem;
		o.mem = 0;
		return *this;
	}

	operator void *() const { return mem; }

	void *operator*() { return mem; }

	operator bool() const { return mem; }

	void *devptr() const {
		static_assert(host, "cannot call devptr() on a device buffer");
		void *ret;
		cudaHostGetDevicePointer((void **) &ret, (void *) mem, 0) && assertcu;
		return ret;
	}

	~cudalist() {
		if (!mem) return;
		host ? cudaFreeHost((void *) mem) : cudaFree((void *) mem);
		mem = 0;
	}

private:
	void *mem = 0;
};

/*
 * completion represents a time point in a stream. It is used to easily synchronize between streams,
 * and to make the host wait on stream completions. This is essentially a wrapper struct around cudaEvent_t.
 */

struct completion {

	completion() = default;

	explicit completion(cudaStream_t stream) {
		record(stream);
	}

	completion(completion &&o) {
		*this = std::move(o);
	}

	completion &operator=(completion &&o) {
		delevent();
		std::swap(o.event, event);
		return *this;
	}

	~completion() {
		delevent();
	}

	completion &record(cudaStream_t stream) {
		newevent();
		cudaEventRecord(event, stream) && assertcu;
		return *this;
	}

	void blocks(cudaStream_t stream) const {
		if (!event) return;
		cudaStreamWaitEvent(stream, event, 0) && assertcu;
	}

	void wait() const {
		if (!event) return;
		cudaEventSynchronize(event) && assertcu;
	}

	bool ready() const {
		if (!event) return true;
		auto ret = cudaEventQuery(event);
		if (ret == cudaErrorNotReady) return false;
		ret && assertcu;
		return true;
	}

private:
	void delevent() {
		if (!event) return;
		cudaEventDestroy(event);
		event = 0;
	}

	void newevent() {
		delevent();
		cudaEventCreateWithFlags(&event, cudaEventDisableTiming) && assertcu;
	}

	cudaEvent_t event = 0;
};

/*
 * Lambda -> cudaCallback transform. All exceptions are caught and set on the provided
 * exception_ptr reference, to be read by your main thread.
 */

#include <iostream>
#include <exception>
#include <memory>

template<typename L>
struct cuda_callback_data {
	std::exception_ptr &callback_err;
	L lambda;
};

template<typename L>
void add_cuda_callback(cudaStream_t stream, std::exception_ptr &callback_err, L &lambda) {
	cudaStreamAddCallback(stream, +[](cudaStream_t stream, cudaError_t status, void *userData) {
		std::unique_ptr<cuda_callback_data<L &>> data(reinterpret_cast<cuda_callback_data<L &> *>(userData));
		try { data.lambda(status); }
		catch (...) { data.callback_err = std::current_exception(); }
	}, new cuda_callback_data<L &>{callback_err, lambda}, 0) && assertcu;
}

template<typename L>
void add_cuda_callback(cudaStream_t stream, std::exception_ptr &callback_err, L &&lambda) {
	cudaStreamAddCallback(stream, +[](cudaStream_t stream, cudaError_t status, void *userData) {
		std::unique_ptr<cuda_callback_data<L>> data(reinterpret_cast<cuda_callback_data<L> *>(userData));
		try { data->lambda(status); }
		catch (...) { data->callback_err = std::current_exception(); }
	}, new cuda_callback_data<L>{callback_err, std::move(lambda)}, 0) && assertcu;
}
