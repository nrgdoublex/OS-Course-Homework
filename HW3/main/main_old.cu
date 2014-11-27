#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <cuda.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

//page size is 32 bytes
#define PAGESIZE (32)
//32 KB in the shared memory
#define PHYSICAL_MEM_SIZE (32768)
//128 KB of secondary storage
#define STORAGE_SIZE (131072)

//number of pages in shared memory
#define PHYSICAL_PAGE_NUM (PHYSICAL_MEM_SIZE/PAGESIZE)
//number of pages in global memory
#define STORAGE_PAGE_NUM (STORAGE_MEM_SIZE/PAGESIZE)

#define DATAFILE "./data.bin"
#define OUTFILE "./snapshot.bin"
typedef unsigned char uchar;
typedef uint32_t u32;

#define VALID_BIT 						(1)
#define INVALID_BIT						(0)

#define IN_STORAGE						(1)
#define NOT_IN_STORAGE					(0)

#define VALID_BIT_START 				(0)
#define VALID_BIT_LEN					(1)
#define STORAGE_BIT_START 				(VALID_BIT_START+VALID_BIT_LEN)
#define STORAGE_BIT_LEN 				(1)	
#define MEM_PAGE_START				 	(STORAGE_BIT_START+STORAGE_BIT_LEN)
#define MEM_PAGE_LEN				 	(10)
#define COUNTER_START					(MEM_PAGE_START+MEM_PAGE_LEN)
#define COUNTER_LEN						(10)

#define VALID_BIT_MASK					((1<<(VALID_BIT_START+VALID_BIT_LEN))-1)
#define STORAGE_BIT_MASK				((1<<(STORAGE_BIT_START+STORAGE_BIT_LEN))-1)
#define MEM_PAGE_MASK					((1<<(MEM_PAGE_START+MEM_PAGE_LEN))-1)
#define COUNTER_MASK					((1<<(COUNTER_START+COUNTER_LEN))-1)

#define GET_VALID_BIT(x)				((x&VALID_BIT_MASK)>>VALID_BIT_START)
#define GET_STORAGE_BIT(x)				((x&STORAGE_BIT_MASK)>>STORAGE_BIT_START)
#define GET_MEM_PAGE(x)					((x&MEM_PAGE_MASK)>>MEM_PAGE_START)
#define GET_COUNTER(x)					((x&COUNTER_MASK)>>COUNTER_START)

#define CLEAR_VALID_BIT(x)				(x&(~(((1<<VALID_BIT_LEN)-1)<<VALID_BIT_START)))
#define CLEAR_STORAGE_BIT(x)			(x&(~(((1<<STORAGE_BIT_LEN)-1)<<STORAGE_BIT_START)))
#define CLEAR_MEM_PAGE_BITS(x)			(x&(~(((1<<MEM_PAGE_LEN)-1)<<MEM_PAGE_START)))
#define CLEAR_COUNTER_BITS(x)			(x&(~(((1<<COUNTER_LEN)-1)<<COUNTER_START)))

#define SET_VALID_BIT(src)				(src|(VALID_BIT<<VALID_BIT_START))
#define SET_STORAGE_BIT(src)			(src|(IN_STORAGE<<STORAGE_BIT_START))
#define SET_MEM_PAGE(src,value)			(CLEAR_MEM_PAGE_BITS(x)|(value<<MEM_PAGE_START))
#define SET_COUNTER(src,value)			(CLEAR_COUNTER_BITS(x)|(value<<COUNTER_START))

//page table entries
__device__ __managed__ int PAGE_ENTRIES = 0;
//Page-fault times
__device__ __managed__ int PAGEFAULT = 0;

//secondary memory
__device__ __managed__ uchar storage[STORAGE_SIZE];

//data input and output
__device__ __managed__ uchar results[STORAGE_SIZE];
__device__ __managed__ uchar input[STORAGE_SIZE];

//shared memory for page table and memory-occupied table
extern __shared__ u32 pt[];

//point to next available memory page
__device__ __managed__ int NEXT_AVAIL_PAGE;

__device__ u32 paging(uchar *buffer, u32 page_num, u32 offset)
{
	u32 addr;
	u32 valid 		= GET_VALID_BIT(pt[page_num]);
	u32 instorage 	= GET_STORAGE_BIT(pt[page_num]);
	u32 mempage 	= GET_MEM_PAGE(pt[page_num]);
	u32 counter 	= GET_COUNTER(pt[page_num]);

	//if valid bit is not set and it is not in storage
	if(!valid && !instorage){
		if(NEXT_AVAIL_PAGE<PHYSICAL_PAGE_NUM){
			addr = NEXT_AVAIL_PAGE*PAGESIZE+offset;

			SET_VALID_BIT(pt[page_num]);
			SET_MEM_PAGE(pt[page_num],NEXT_AVAIL_PAGE);
			SET_COUNTER(pt[page_num],0);

			NEXT_AVAIL_PAGE++;
			for(int i=0;i<STORAGE_PAGE_NUM;i++){
				if(GET_VALID_BIT(pt[i]) && i!=page_num)
					SET_COUNTER(pt[i],GET_COUNTER(i)+1);
			}
		}
		else{
			int largest_value=0;
			int largest_idx=0;
			int shared_start_addr,storage_start_addr;
			for(int i=0;i<STORAGE_PAGE_NUM;i++){
				if(GET_VALID_BIT(pt[i]) && GET_COUNTER(pt[i])> largest_value){
					largest_value = GET_COUNTER(pt[i]);
					largest_idx=i;
				}
			}
			shared_start_addr = GET_MEM_PAGE(pt[largest_idx])*PAGE_SIZE;
			storage_start_addr = largest_idx*PAGESIZE;
			for(int i=0;i<PAGESIZE;i++){
				storage[storage_start_addr+i]=buffer[shared_start_addr+i];
				buffer[shared_start_addr+i]=0;
			}
			
			addr = shared_start_addr + offset;
			SET_VALID_BIT(pt[page_num]);
			SET_MEM_PAGE(pt[page_num],GET_MEM_PAGE(largest_idx));
			SET_COUNTER(pt[page_num],0);

			CLEAR_VALID_BIT(pt[largest_idx]);
			SET_STORAGE_BIT(pt[largest_idx]);
		}
	}
	//if valid bit is not set and it is in storage
	else if(!valid){
		
	}
	//if valid bit is set
	else if(valid){
	}
	return addr;
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
	u32 offset	  = addr%PAGESIZE;

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
	int pt_entries = STORAGE_SIZE/PAGESIZE;
	NEXT_AVAIL_PAGE=0;
	for(int i=0;i<pt_entries;i++){
		pt[i]= pt[i] && (~pt[i]);
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
	int pt_entries = STORAGE_SIZE/PAGESIZE;

	//We should initialize the page table
	init_pageTable(pt_entries);

	//####GWrite/Gread code section start####
	for(int i=0;i<input_size;i++)
		Gwrite(data,i,input[i]);

	for(int i=input_size-1;i>=input_size -10;i--)
		int value = Gread(data,i);

	snapshot(results, data, 0, input_size);
	//####GWrite/Gread code section end####
	printf("pagefault times = %d\n",PAGEFAULT);
}

int main()
{
	//Load data.bin into input buffer
	//int input_size = load_binaryFile(DATAFILE, input, STORAGE_SIZE);

	//printf("The read size is %d\n", input_size);

	//main procedure
	//cudaSetDevice(1);
	//mykernel<<<1,1,16384>>>(input_size);
	//cudaDeviceSynchronize();
	//cudaDevicereset();

	//write binary file from results buffer
	//write_binaryFile(OUTFILE, results, input_size);

	//int output_size;
	//output_size=write_binaryFile(OUTFILE, input, input_size);
	//printf("The write size is %d\n",output_size);


	return 0;
}
