#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <cuda.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>

//page size is 32 bytes
#define PAGESIZE (32)
//32 KB in the shared memory
#define PHYSICAL_MEM_SIZE (32768)
//128 KB of secondary storage
#define STORAGE_SIZE (131072)
//memory segment size for each thread
#define MEMORY_SEGMENT (32768)

//number of pages in shared memory
#define PHYSICAL_PAGE_NUM (PHYSICAL_MEM_SIZE/PAGESIZE)
//number of pages in global memory
#define STORAGE_PAGE_NUM (STORAGE_MEM_SIZE/PAGESIZE)

#define DATAFILE "./data.bin"
#define OUTFILE "./snapshot.bin"
typedef unsigned char uchar;
typedef uint32_t u32;

#define VIRPAGE_OFFSET (0)
#define COUNTER_OFFSET (1)
//#define PID_OFFSET     (2)
#define PT_ENTRY_LEN   (2)

#define INVALID_VALUE (0xffffffff)

#define __LOCK(); for(int p=0;p<4;p++){if(threadIdx.x==p){
#define __UNLOCK(); }__syncthreads();}

//#define __GET_BASE() ((threadIdx.x)*MEMORY_SEGMENT)
#define __GET_BASE() (p*MEMORY_SEGMENT)

//page table entries
__device__ __managed__ int PAGE_ENTRIES = 0;
//Page-fault times
__device__ __managed__ u32 PAGEFAULT = 0;

//secondary memory
__device__ __managed__ uchar storage[STORAGE_SIZE];

//data input and output
__device__ __managed__ uchar results[STORAGE_SIZE];
__device__ __managed__ uchar input[STORAGE_SIZE];

//shared memory for page table and memory-occupied table
extern __shared__ u32 pt[];


__device__ u32 paging(uchar *buffer, u32 page_num, u32 offset)
{
	u32 free_page_num=INVALID_VALUE;
	u32 lru_time=1;
	u32 lru_page_num=INVALID_VALUE;
	u32 valid_page_num=INVALID_VALUE;
	for(int i=0;i<PHYSICAL_PAGE_NUM;i++){
		if(pt[i*PT_ENTRY_LEN+VIRPAGE_OFFSET]==page_num){
			pt[i*PT_ENTRY_LEN+COUNTER_OFFSET]=1;
			valid_page_num=i;
		}
		else if(pt[i*PT_ENTRY_LEN+COUNTER_OFFSET]>0)
			pt[i*PT_ENTRY_LEN+COUNTER_OFFSET]++;
		if(pt[i*PT_ENTRY_LEN+VIRPAGE_OFFSET]==INVALID_VALUE)
			free_page_num=i;
		if(pt[i*PT_ENTRY_LEN+COUNTER_OFFSET]>lru_time){
			lru_time=pt[i*PT_ENTRY_LEN+COUNTER_OFFSET];
			lru_page_num=i;
		}
	}
	if(valid_page_num!=INVALID_VALUE){
		return valid_page_num*PAGESIZE+offset;
	}
	else if(free_page_num!=INVALID_VALUE){
		pt[free_page_num*PT_ENTRY_LEN+VIRPAGE_OFFSET]=page_num;
		pt[free_page_num*PT_ENTRY_LEN+COUNTER_OFFSET]=1;
		PAGEFAULT++;
		return free_page_num*PAGESIZE+offset;
	}
	else{
		u32 swap_out_start=pt[lru_page_num*PT_ENTRY_LEN+VIRPAGE_OFFSET]*PAGESIZE;
		u32 swap_in_start=page_num*PAGESIZE;
		u32 phy_start=lru_page_num*PAGESIZE;
		for(int i=0;i<PAGESIZE;i++){
			storage[swap_out_start+i]=buffer[phy_start+i];
			buffer[phy_start+i]=storage[swap_in_start+i];
		}
		pt[lru_page_num*PT_ENTRY_LEN+VIRPAGE_OFFSET]=page_num;
		pt[lru_page_num*PT_ENTRY_LEN+COUNTER_OFFSET]=1;
		PAGEFAULT++;
		return phy_start+offset; 
	}

}

__device__ uchar Gread(uchar *buffer,u32 addr)
{
	u32 page_num  = addr/PAGESIZE;
	u32 offset 	  = addr%PAGESIZE;

	//addr means the addr in shared memory
	addr = paging(buffer, page_num, offset);
	return buffer[addr];
}
__device__ void Gwrite(uchar *buffer, u32 addr, uchar value)
{
	u32 page_num  = addr/PAGESIZE;
	u32 offset    = addr%PAGESIZE;

	//addr means the addr in shared memory
	addr = paging(buffer, page_num, offset);
	buffer[addr] = value;
}

__device__ void snapshot(uchar *results, uchar *buffer, int offset, int input_size)
{
	for(int i=0;i<input_size;i++)
		results[i] = Gread(buffer, i+offset);
}

__device__ void init_pageTable(int pt_entries)
{
	PAGEFAULT=0;
	for(int i=0;i<PHYSICAL_PAGE_NUM*PT_ENTRY_LEN;i+=PT_ENTRY_LEN){
		pt[i+VIRPAGE_OFFSET]=INVALID_VALUE;
		pt[i+COUNTER_OFFSET]=0;
	}
	for(int i=0;i<STORAGE_SIZE;i++){
		results[i]=0;
		storage[i]=0;
	}
}

int load_binaryFile(const char *filename, uchar *input, int size)
{
	int fd=0;
	int sizeread=0;
	int sizehasread=0;
	fd=open(filename,O_RDONLY);
	if(fd==-1){
		perror("open data.bin error");
		return -1;
	}
	while((sizeread=read(fd,input,size))!=-1){
		sizehasread+=sizeread;
		size-=sizeread;
		input+=sizeread;
		if(sizeread==0){
			close(fd);
			return sizehasread;
		}
	}
	close(fd);
	perror("read data.bin error");
	return -1;
}

int write_binaryFile(const char *filename, uchar *output, int size)
{
	int fd=0;
	int sizewritten=0;
	int sizehaswritten=0;
	fd=open(filename,O_WRONLY|O_CREAT,S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH);
	if(fd==-1){
		perror("open snapshot.bin error");
		return -1;
	}
	while((sizewritten=write(fd,output,size))!=-1){
		sizehaswritten+=sizewritten;
		size-=sizewritten;
		output+=sizewritten;
		if(sizewritten==0){
			close(fd);
			return sizehaswritten;
		}
	}
	close(fd);
	perror("write snapshot.bin error");
	return -1;
}

__global__ void mykernel(int input_size)
{
	//Regard shared memory as physical memory
	__shared__ uchar data[PHYSICAL_MEM_SIZE];

	//get page table entries
	int pt_entries;
	pt_entries = PHYSICAL_MEM_SIZE/PAGESIZE;

	//We should initialize the page table
	if(threadIdx.x==0)
		init_pageTable(pt_entries);

	
	//####GWrite/Gread code section start####
	__LOCK();
	for(int i=0;i<input_size;i++)
		Gwrite(data,i+__GET_BASE(),input[i+__GET_BASE()]);
	__UNLOCK();


	for(int i=input_size-1;i>=input_size-10;i--){
	__LOCK();
		int value = Gread(data,i+__GET_BASE());
	__UNLOCK();
	}

	__LOCK();
	snapshot(results+__GET_BASE(),data,__GET_BASE(),input_size);
	__UNLOCK();
	//####GWrite/Gread code section end####
	printf("thread number = %d, pagefault times=%u\n",threadIdx.x,PAGEFAULT);
	return;
}

__global__ void hello()
{
	printf("hello, world!!\n");
}

int main()
{
	clock_t t;
	t=clock();

	//Load data.bin into input buffer
	int input_size = load_binaryFile(DATAFILE, input, STORAGE_SIZE);

	printf("The read size is %d\n", input_size);

	//main procedure
	cudaSetDevice(4);
	mykernel<<<1,4,16384>>>(input_size/4);
	cudaDeviceSynchronize();
	cudaDeviceReset();

	//write binary file from results buffer
	write_binaryFile(OUTFILE, results, input_size);

	t=clock()-t;
	printf("total elapsed time = %f\n",((float)t)/CLOCKS_PER_SEC);
	//int output_size;
	//output_size=write_binaryFile(OUTFILE, input, input_size);
	//printf("The write size is %d\n",output_size);

	return 0;
}
