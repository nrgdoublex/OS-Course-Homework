#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <cuda.h>

//total size of available memory
#define STORAGE_SIZE 		(1085440)
//the max size of file memory
#define FILE_STORAGE_SIZE 	(1048576)
//the max size of a file block
#define BLOCK_SIZE			(1024)
//the start position of file memory
#define FILE_STORAGE_START	(STORAGE_SIZE-FILE_STORAGE_SIZE)

//input file
#define DATAFILE "./data.bin"
//output file
#define OUTFILE "./snapshot.bin"
typedef unsigned char uchar;
typedef uint32_t u32;
typedef uint16_t u16;

#define INVALID_VALUE		(0xffffffff)
#define FILE_OPEN_ERROR		INVALID_VALUE
#define WRITE_ERROR			INVALID_VALUE
#define READ_ERROR			INVALID_VALUE

#define TRUE				(1)
#define FALSE				(0)

#define VALID				(1)
#define FREE				(0)
#define INVALID				(0)

#define FREE_BLOCK_MASK		(0x1)
#define FREE_BLOCK_BIT		(0)
#define NON_FREE_BLOCK_MASK	(1)

//used in the bit marking whether the FCB stores file or directory
#define FILE_BIT			(0)
#define DIR_BIT				(1)

//used as read/write flag
#define G_READ				(0)
#define G_WRITE				(1)

//max number of files
#define MAX_FILE_NUM		(FILE_STORAGE_SIZE/BLOCK_SIZE)
#define BITS_IN_BYTE		(8)

//max size of file
#define MAX_FILE_SIZE		(1024)

//max size of file name
#define MAX_FILE_NAME		(20)

//gsys flags
#define RM					(0)
#define LS_D				(1)
#define LS_S				(2)

typedef struct{
	uchar bitmap[MAX_FILE_NUM/BITS_IN_BYTE];
	u32 current_dir			:10;
	u32 file_num			:10;
	u32 valid_current_dir	:1;
	u16 file_list_time[MAX_FILE_NUM];
	u16 file_list_size[MAX_FILE_NUM];
}SystemInfo;

typedef struct{
	char name[MAX_FILE_NAME];
	u32 valid_entry		:1;
	u32 dir_bit			:1;
	u32 mode			:1;
	u32 create_time		:10;
	u32 modified_time	:10;
	u32 block_num		:10;
	u32 file_size		:10;
	u32 parent_dir		:10;
	u32 has_parent_dir	:1;
}FCB;

//FCB array pointer
__device__ __managed__ FCB *fcb_table;

//Super block pointer
__device__ __managed__ SystemInfo *system_info;

//total storage 
__device__ __managed__ uchar *volume;

__device__ u32 get_bitmap(u32 index)
{
	u32 byte_num = index/BITS_IN_BYTE;
	u32 offset	 = index%BITS_IN_BYTE;
	return ((system_info->bitmap[byte_num])>>offset)&FREE_BLOCK_MASK;
}

__device__ void set_bitmap(u32 index, u32 flag)
{
	u32 byte_num = index/BITS_IN_BYTE;
	u32 offset	 = index%BITS_IN_BYTE;
	if(flag==VALID)
		system_info->bitmap[byte_num] = system_info->bitmap[byte_num]|(VALID<<offset);
	else
		system_info->bitmap[byte_num] = system_info->bitmap[byte_num]|(FREE<<offset);
}

__device__ u32 cmp_filename(const char *dest,const char *src)
{
	u32 index=0;
	while(index<MAX_FILE_NAME){
		if(src[index]!=dest[index])
			return 1;
		else if(src[index]=='\0' && dest[index]=='\0')
			return 0;
		index++;
	}
	return 0;
}

__device__ u32 cpy_filename(char *dest,const char *src)
{
	u32 index=0;
	while(src[index]!='\0' && index<20){
		dest[index]=src[index];
		index++;
	}
	dest[++index]='\0';
	return index;
}

/************************************************************************************
 *
 * This is main function opening a file
 *
 ************************************************************************************/
__device__ u32 open(const char *name, u32 mode)
{
	u32 file_idx=INVALID_VALUE;
	u32 free_fcb=INVALID_VALUE;
	//search for FCB by file name
	for(int i=0;i<MAX_FILE_NUM;i++){
		if(cmp_filename(fcb_table[i].name,name)==0&&fcb_table[i].valid_entry==VALID){
			file_idx = i;
			break;
		}
		if(fcb_table[i].valid_entry==FREE)
			free_fcb=i;
	}
	//if found, return address
	if(file_idx!=INVALID_VALUE){
		fcb_table[file_idx].mode=mode;
		return file_idx;
	}
	//if not found
	else{
		u32 free_block=INVALID_VALUE;
		//search the bitmap for free block
		for(int i=0;i<MAX_FILE_NUM;i++){
			if(get_bitmap(i)==FREE_BLOCK_BIT){
				free_block=i;
				break;
			}
		}
		//if there is a free block, create a new FCB and record its block number
		if(free_block!=INVALID_VALUE){
			//renew modified time 
			for(int i=0;i<MAX_FILE_NUM;i++){
				if(fcb_table[i].valid_entry==VALID)
					fcb_table[i].modified_time++;
					fcb_table[i].create_time++;
			}
			fcb_table[free_fcb].modified_time=0;
			fcb_table[free_fcb].create_time=0;
			//set the mode of file
			fcb_table[free_fcb].mode=mode;
			//set the FCB entry valid
			fcb_table[free_fcb].valid_entry=TRUE;
			//make FCB point to the free block
			fcb_table[free_fcb].block_num=free_block;
			//set the file name
			cpy_filename(fcb_table[free_fcb].name,name);
			//set bitmap to indicate it is occupied
			set_bitmap(free_block,VALID);
			//renew total number of files
			system_info->file_num++;

			//renew file lists
			system_info->file_list_time[system_info->file_num-1]=free_fcb;
			system_info->file_list_size[system_info->file_num-1]=free_fcb;

			return free_fcb;
		}
		//if no free blocks are available, return error;
		else{
			return FILE_OPEN_ERROR;
		}
	}
}

/************************************************************************************
 *
 * This is main function removing a file
 *
 ************************************************************************************/
__device__ void rm(const char *filename)
{
	u32 file_fcb=INVALID_VALUE;
	//search for FCB by file name
	for(int i=0;i<MAX_FILE_NUM;i++){
		if(cmp_filename(fcb_table[i].name,filename)==0){
			file_fcb = i;
			break;
		}
	}
	if(file_fcb!=INVALID_VALUE){
		u32 file_start = fcb_table[file_fcb].block_num*BLOCK_SIZE;
		u32 time = fcb_table[file_fcb].modified_time;
		u32 create_time = fcb_table[file_fcb].create_time;
		u16 flag=FALSE;
		//remove file in file list
		for(int i=0;i<system_info->file_num;i++){
			if(system_info->file_list_time[i]==file_fcb)
				flag=TRUE;
			if(flag==TRUE && i!=system_info->file_num-1)
				system_info->file_list_time[i]=system_info->file_list_time[i+1];
			else if(flag==TRUE)
				system_info->file_list_time[i]=0;
		}
		//reset system info
		system_info->file_num--;
		set_bitmap(fcb_table[file_fcb].block_num, FREE);
		//reset modified time of other files
		for(int i=0;i<MAX_FILE_NUM;i++){
			if(fcb_table[i].valid_entry==VALID){
				if(fcb_table[i].modified_time>time)
					fcb_table[i].modified_time--;
				else if(fcb_table[i].create_time>create_time)
					fcb_table[i].create_time--;
			}
		}
		//clear file content
		for(int i=0;i<MAX_FILE_SIZE;i++)
			volume[FILE_STORAGE_START+file_start+i]=0;
		//reset FCB block
		fcb_table[file_fcb].valid_entry		=FREE;
		fcb_table[file_fcb].dir_bit			=FILE_BIT;
		fcb_table[file_fcb].mode			=G_READ;
		fcb_table[file_fcb].create_time		=0;
		fcb_table[file_fcb].modified_time	=0;
		fcb_table[file_fcb].block_num		=0;
		fcb_table[file_fcb].file_size		=0;
		fcb_table[file_fcb].parent_dir		=0;
		fcb_table[file_fcb].has_parent_dir	=FALSE;
		for(int j=0;j<MAX_FILE_NAME;j++)
			fcb_table[file_fcb].name[j]=0;

	}
	else{
		printf("Cannot find file %s\n",filename);
	}
}

/************************************************************************************
 *
 * This is main function writing a file
 *
 ************************************************************************************/
__device__ u32 write(const uchar *input, u32 num, u32 fp)
{
	//if file is not in write mode, return error
	if(fcb_table[fp].mode!=G_WRITE){
		printf("%s is not in write mode\n",fcb_table[fp].name);
		return WRITE_ERROR;
	}
	u32 file_start		= fcb_table[fp].block_num*BLOCK_SIZE;
	u32 previous_time	= fcb_table[fp].modified_time;
	u32 count			=(num<MAX_FILE_SIZE)?num:MAX_FILE_SIZE;

	//if bytes to write is more than max file size
	if(num>MAX_FILE_SIZE)
		printf("Cannot write more than 1024 bytes in a file\n");
	//write the file
	for(int i=0;i<count;i++)
		volume[FILE_STORAGE_START+file_start+i]=input[i];
	//renew the file size in FCB
	fcb_table[fp].file_size=count;
	//rwnew modified time in FCB
	for(int i=0;i<MAX_FILE_NUM;i++){
		if(fcb_table[i].valid_entry==VALID && fcb_table[i].modified_time <= previous_time){
			fcb_table[i].modified_time++;
		}
	}
	fcb_table[fp].modified_time=0;

	//renew file lists
	for(int i=0;i<system_info->file_num-1;i++){
		if(fcb_table[system_info->file_list_time[i]].modified_time<=previous_time)
			system_info->file_list_time[i]=system_info->file_list_time[i+1];
	}
	system_info->file_list_time[system_info->file_num-1]=fp;

	u32 start_idx, end_idx;
	u16 flag=FALSE;
	for(int i=0;i<system_info->file_num;i++){
		if(system_info->file_list_size[i]==fp)
			start_idx=i;
		if(fcb_table[system_info->file_list_size[i]].file_size<=count && \
			fcb_table[system_info->file_list_size[i]].create_time>=fcb_table[fp].create_time && \
			flag==FALSE){
			flag=TRUE;
			end_idx=i;
		}
	}
	if(end_idx>start_idx){
		u32 temp=system_info->file_list_size[start_idx];
		for(int i=start_idx;i<end_idx;i++)
			system_info->file_list_size[i]=system_info->file_list_size[i+1];
		system_info->file_list_size[end_idx]=temp;
	}
	else{
		u32 temp=system_info->file_list_size[start_idx];
		for(int i=start_idx;i>end_idx;i--)
			system_info->file_list_size[i]=system_info->file_list_size[i-1];
		system_info->file_list_size[end_idx]=temp;
	}
	
	//return number of bytes written
	return count;
}

/************************************************************************************
 *
 * This is main function reading a file
 *
 ************************************************************************************/
__device__ u32 read(uchar *output, u32 num, u32 fp)
{
	//if file is not in read mode, return error
	if(fcb_table[fp].mode!=G_READ){
		printf("%s is not in read mode\n",fcb_table[fp].name);
		return READ_ERROR;
	}

	u32 file_start = fcb_table[fp].block_num*BLOCK_SIZE;
	u32 count = (num<fcb_table[fp].file_size)?num:fcb_table[fp].file_size;
	//if bytes to read is more than max file size
	if(num>fcb_table[fp].file_size)
		printf("Cannot read more than file size\n");
	//read the file
	for(int i=0;i<count;i++)
		output[i]=volume[FILE_STORAGE_START+file_start+i];
	//return number of bytes read
	return count;
}

/************************************************************************************
 *
 * This is main function listing a file by modified time
 *
 ************************************************************************************/
__device__ void list_file_time()
{
	for(int i=system_info->file_num-1;i>=0;i--)
		printf("%s\n",fcb_table[system_info->file_list_time[i]].name);
}

/************************************************************************************
 *
 * This is main function listing a file by file size
 *
 ************************************************************************************/
__device__ void list_file_size()
{
	for(int i=0;i<system_info->file_num;i++)
		printf("%s\n",fcb_table[system_info->file_list_size[i]].name);
}

__device__ void gsys(u32 flag, const char *filename)
{
	switch(flag){
		case RM:
			rm(filename);
			break;
		default:
			printf("Please enter correct command\n");
	}
}

__device__ void gsys(u32 flag)
{
	switch(flag){
		case LS_D:
			printf("===sort by modified time===\n");
			list_file_time();
			break;
		case LS_S:
			printf("===sort by file size===\n");
			list_file_size();
			break;
		default:
			printf("Please enter correct command\n");
	}
}

int load_binaryFile(const char *filename, uchar *input, int size)
{
	FILE *fd;
	int sizeread=0;
	fd=fopen(filename,"r");
	if(fd==NULL){
		perror("open data.bin error");
		return -1;
	}
	sizeread = fread(input, sizeof(uchar), size, fd);
	if(sizeread!=size){
		printf("read %s error\n",filename);
		fclose(fd);
		return -1;
	}
	fclose(fd);
	printf("read_size = %d\n",sizeread);
	return 0;
}

int write_binaryFile(const char *filename, uchar *output, int size)
{
	FILE *fd;
	int sizewritten=0;
	fd=fopen(filename,"w");
	if(fd==NULL){
		perror("open snapshot.bin error");
		return -1;
	}
	sizewritten=fwrite(output, sizeof(uchar), size, fd);
	if(sizewritten!=size){
		printf("write %s error\n",filename);
		fclose(fd);
		return -1;
	}
	fclose(fd);
	printf("write_size = %d\n",sizewritten);
	return 0;
}	

void init_volume()
{
	system_info = (SystemInfo *)volume;
	fcb_table = (FCB *)(volume+sizeof(*system_info));
	
	for(int i=0;i<MAX_FILE_NUM/BITS_IN_BYTE;i++)
		system_info->bitmap[i]=0;
	system_info->valid_current_dir=INVALID;
	system_info->current_dir=INVALID;
	system_info->file_num=0;
	for(int i=0;i<MAX_FILE_NUM;i++){
		system_info->file_list_time[i]=0;
		system_info->file_list_size[i]=0;
	}

	for(int i=0;i<MAX_FILE_NUM;i++){
		fcb_table[i].valid_entry	=FREE;
		fcb_table[i].dir_bit		=FILE_BIT;
		fcb_table[i].mode			=G_READ;
		fcb_table[i].create_time	=0;
		fcb_table[i].modified_time	=0;
		fcb_table[i].block_num		=0;
		fcb_table[i].file_size		=0;
		fcb_table[i].parent_dir		=0;
		fcb_table[i].has_parent_dir	=FALSE;
		for(int j=0;j<MAX_FILE_NAME;j++){
			fcb_table[i].name[j]=0;
		}
	}
}

__global__ void mykernel(uchar *input, uchar *output)
{
	uchar str[20]="this is a pen";
	uchar str1[20]="this is not a pen";
	uchar out[20];
	//####kernel start####
	u32 fp=open("t.txt\0",G_WRITE);
	write(str, 14, fp);
	u32 fp3=open("a.txt\0",G_WRITE);
	u32 fp4=open("b.txt\0",G_WRITE);
	u32 fp5=open("t.txt\0",G_WRITE);
	write(str1,18,fp5);
	u32 fp2=open("t.txt\0",G_READ);
	read(out, 18, fp2);
	u32 fp6=open("c.txt\0",G_WRITE);
	u32 fp7=open("b.txt\0",G_WRITE);
	write(str,14,fp7);
	gsys(LS_S);
//	rm("b.txt\0");
//	gsys(LS_D);
	//####kernel end####
	printf("file num = %d\n",system_info->file_num);
//	printf("the string is %s\n",out);

	printf("create time = %d\n",fcb_table[1023].create_time);
	printf("name = %s\n", fcb_table[1023].name);
	printf("create time = %d\n",fcb_table[1022].create_time);
	printf("name = %s\n", fcb_table[1022].name);
	printf("create time = %d\n",fcb_table[1021].create_time);
	printf("name = %s\n", fcb_table[1021].name);
	printf("create time = %d\n",fcb_table[1020].create_time);
	printf("name = %s\n", fcb_table[1020].name);
}

int main()
{
	cudaMallocManaged(&volume, STORAGE_SIZE);
	init_volume();

	uchar *input, *output;
	cudaMallocManaged(&input, FILE_STORAGE_SIZE);
	cudaMallocManaged(&output, FILE_STORAGE_SIZE);
	for(int i=0;i<FILE_STORAGE_SIZE;i++)
		output[i]=0;

	load_binaryFile(DATAFILE, input, FILE_STORAGE_SIZE);
	
	cudaSetDevice(4);
	mykernel<<<1,1>>>(input, output);

	cudaDeviceSynchronize();
	write_binaryFile(OUTFILE, output, FILE_STORAGE_SIZE);
	cudaDeviceReset();
	
	printf("FCB size = %ld\n",sizeof(FCB));
//	printf("volume start = %ld\n",(long unsigned int)volume);
//	printf("system_info start = %ld\n",(long unsigned int)system_info);
//	printf("FCB start = %ld\n",(long unsigned int)fcb_table);
//	printf("Storage start = %ld\n",(long unsigned int)&volume[FILE_STORAGE_START]);
//	printf("%d\n",fcb_table[100].file_size);
//	printf("%d\n",system_info->file_num);
	return 0;
}
