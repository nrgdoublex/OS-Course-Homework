#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <cuda.h>
#include <string.h>

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

//for valid/invalid FCB and valid/free block
#define VALID				(1)
#define FREE				(0)
#define INVALID				(0)

//for bitmap
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

//storing syatem information
typedef struct{
	uchar bitmap[MAX_FILE_NUM/BITS_IN_BYTE];	//the bitmap recording free blocks
	u32 file_num			:11;		//the number of files
	u16 file_list_time[MAX_FILE_NUM];	//list files in order of decreasing modified time
	u16 file_list_size[MAX_FILE_NUM];	//list files in order of decreasing file size
}SystemInfo;

//FCB entry
typedef struct{
	char name[MAX_FILE_NAME];	//file name
	u32 valid_entry		:1;		//indicate whether this entry is valid
	u32 mode			:1;		//the R/W mode of the file
	u32 create_time		:10;	//the time when the file is created
	u32 modified_time	:10;	//the last modified time of a file
	u32 block_num		:10;	//the index of its file block
	u32 file_size		:11;	//the size of a file
}FCB;

//FCB array pointer
__device__ __managed__ FCB *fcb_table;

//system struct pointer
__device__ __managed__ SystemInfo *system_info;

//total storage 
__device__ __managed__ uchar *volume;

/************************************************************************************
 *
 * Get bitmap value
 *
 ************************************************************************************/
__device__ u32 get_bitmap(u32 index)
{
	u32 byte_num = index/BITS_IN_BYTE;
	u32 offset	 = index%BITS_IN_BYTE;
	return ((system_info->bitmap[byte_num])>>offset)&FREE_BLOCK_MASK;
}

/************************************************************************************
 *
 * Set bitmap value
 *
 ************************************************************************************/
__device__ void set_bitmap(u32 index, u32 flag)
{
	u32 byte_num = index/BITS_IN_BYTE;
	u32 offset	 = index%BITS_IN_BYTE;
	if(flag==VALID)
		system_info->bitmap[byte_num] = system_info->bitmap[byte_num]|(VALID<<offset);
	else{
		system_info->bitmap[byte_num] = system_info->bitmap[byte_num]&(~(VALID<<offset));
//		system_info->bitmap[byte_num] = system_info->bitmap[byte_num]|(FREE<<offset);
	}
}

/************************************************************************************
 *
 * Compare file names in the kernel function
 *
 ************************************************************************************/
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

/************************************************************************************
 *
 * Copy file names
 *
 ************************************************************************************/
__device__ u32 cpy_filename(char *dest,const char *src)
{
	u32 index=0;
	while(src[index]!='\0' && index<MAX_FILE_NAME){
		dest[index]=src[index];
		index++;
	}
	dest[index]='\0';
	return index;
}

/************************************************************************************
 *
 * This is main function opening a file
 *
 ************************************************************************************/
__device__ u32 open(const char *name, u32 mode)
{
	//the index of the FCB entry of a file
	u32 file_fcb=INVALID_VALUE;
	//the index of a free FCB entry
	u32 free_fcb=INVALID_VALUE;
	//search for FCB by file name
	for(int i=0;i<MAX_FILE_NUM;i++){
		if(cmp_filename(fcb_table[i].name,name)==0 && fcb_table[i].valid_entry==VALID){
			file_fcb = i;
			break;
		}
		if(fcb_table[i].valid_entry==FREE)
			free_fcb=i;
	}
	//if found, set its mode and return address
	if(file_fcb!=INVALID_VALUE){
		fcb_table[file_fcb].mode=mode;
		return file_fcb;
	}
	//if not found
	else{
		//the index of a free block
		u32 free_block=INVALID_VALUE;
		//search the bitmap for free block
		for(int i=0;i<MAX_FILE_NUM;i++){
			if(get_bitmap(i)==FREE){
				free_block=i;
				break;
			}
		}
		//if there is a free block, create a new FCB and record its block number
		if(free_block!=INVALID_VALUE){
			//renew modified time of other files
			for(int i=0;i<MAX_FILE_NUM;i++){
				if(fcb_table[i].valid_entry==VALID)
					fcb_table[i].modified_time++;
					fcb_table[i].create_time++;
			}
			//set the modified and create time to be the newest
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
			printf("no free block\n");
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
	//the FCB entry of the to-be-removed file
	u32 file_fcb=INVALID_VALUE;
	//search for FCB by file name
	for(int i=0;i<MAX_FILE_NUM;i++){
		if(cmp_filename(fcb_table[i].name,filename)==0){
			file_fcb = i;
			break;
		}
	}
	//if found
	if(file_fcb!=INVALID_VALUE){
		//the real position of the file
		u32 file_start = fcb_table[file_fcb].block_num*BLOCK_SIZE;
		//the modified time of the file
		u32 time = fcb_table[file_fcb].modified_time;
		//the created time of the file
		u32 create_time = fcb_table[file_fcb].create_time;
		u16 flag=FALSE;

		//remove file in file list
		for(int i=0;i<system_info->file_num;i++){
			if(system_info->file_list_time[i]==file_fcb)
				flag=TRUE;
			if(flag==TRUE && i!=system_info->file_num-1)
				system_info->file_list_time[i]=system_info->file_list_time[i+1];
			else if(flag==TRUE && i==system_info->file_num-1)
				system_info->file_list_time[i]=0;
		}
		flag=FALSE;
		for(int i=0;i<system_info->file_num;i++){
			if(system_info->file_list_size[i]==file_fcb)
				flag=TRUE;
			if(flag==TRUE && i!=system_info->file_num-1)
				system_info->file_list_size[i]=system_info->file_list_size[i+1];
			else if(flag==TRUE && i==system_info->file_num-1)
				system_info->file_list_size[i]=0;
		}
	
		//reset system info
		system_info->file_num--;
		set_bitmap(fcb_table[file_fcb].block_num, FREE);
		//reset modified time of other files
		for(int i=0;i<MAX_FILE_NUM;i++){
			if(fcb_table[i].valid_entry==VALID){
				if(fcb_table[i].modified_time>time)
					fcb_table[i].modified_time--;
				if(fcb_table[i].create_time>create_time)
					fcb_table[i].create_time--;
			}
		}
		//clear file content
		for(int i=0;i<MAX_FILE_SIZE;i++)
			volume[FILE_STORAGE_START+file_start+i]=0;
		//reset FCB block
		fcb_table[file_fcb].valid_entry		=FREE;
		fcb_table[file_fcb].mode			=G_READ;
		fcb_table[file_fcb].create_time		=0;
		fcb_table[file_fcb].modified_time	=0;
		fcb_table[file_fcb].block_num		=0;
		fcb_table[file_fcb].file_size		=0;
		for(int j=0;j<MAX_FILE_NAME;j++)
			fcb_table[file_fcb].name[j]=0;

	}
	//if not found
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
	//renew modified time in FCB
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

	u32 start_idx=INVALID_VALUE;
	u32 end_idx=system_info->file_num;
	u16 flag=FALSE;
	for(int i=0;i<system_info->file_num;i++){
		if(system_info->file_list_size[i]==fp)
			start_idx=i;
		if((fcb_table[system_info->file_list_size[i]].file_size<count || ( \
			fcb_table[system_info->file_list_size[i]].file_size==count && \
			fcb_table[system_info->file_list_size[i]].create_time<fcb_table[fp].create_time)) && \
			flag==FALSE){
			flag=TRUE;
			end_idx=i;
		}
	}
	//if we don't find the final position, set it the tail of list
	if(end_idx>start_idx){
		u32 temp=system_info->file_list_size[start_idx];
		for(int i=start_idx;i<end_idx-1;i++)
			system_info->file_list_size[i]=system_info->file_list_size[i+1];
		system_info->file_list_size[end_idx-1]=temp;
	}
	else if(end_idx<start_idx){
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
	char *name;
	u32 size=0;
	for(int i=0;i<system_info->file_num;i++){
		name=fcb_table[system_info->file_list_size[i]].name;
		size=fcb_table[system_info->file_list_size[i]].file_size;
		printf("%s %d\n",name,size);
	}
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
	return 0;
}	

void init_volume()
{
	system_info = (SystemInfo *)volume;
	fcb_table = (FCB *)(volume+sizeof(*system_info));
	
	for(int i=0;i<MAX_FILE_NUM/BITS_IN_BYTE;i++)
		system_info->bitmap[i]=0;
	system_info->file_num=0;
	for(int i=0;i<MAX_FILE_NUM;i++){
		system_info->file_list_time[i]=0;
		system_info->file_list_size[i]=0;
	}

	for(int i=0;i<MAX_FILE_NUM;i++){
		fcb_table[i].valid_entry	=FREE;
		fcb_table[i].mode			=G_READ;
		fcb_table[i].create_time	=0;
		fcb_table[i].modified_time	=0;
		fcb_table[i].block_num		=0;
		fcb_table[i].file_size		=0;
		for(int j=0;j<MAX_FILE_NAME;j++){
			fcb_table[i].name[j]=0;
		}
	}
}

__device__ void print_FCB(const char *filename);
__device__ void print_FCB(int block);

/************************************************************************************
 *
 * The kernel function
 *
 ************************************************************************************/
__global__ void mykernel(uchar *input, uchar *output)
{
	//####kernel start####

	u32 fp=open("t.txt\0",G_WRITE);
	write(input,64,fp);
	fp=open("b.txt\0",G_WRITE);
	write(input+32,32,fp);
	fp=open("t.txt\0",G_WRITE);
	write(input+32,32,fp);
	fp=open("t.txt\0",G_READ);
	read(output,32,fp);
	gsys(LS_D);
	gsys(LS_S);
	fp=open("b.txt\0",G_WRITE);
	write(input+64,12,fp);
	gsys(LS_S);
	gsys(LS_D);
	gsys(RM,"t.txt\0");
	gsys(LS_S);

//---------------------------------Case2-----------------------------

	char fname[10][20];
	for(int i=0;i<10;i++){
		fname[i][0] = i + 33;
		for(int j=1;j<19;j++)
			fname[i][j] = 64+j;
		fname[i][19]='\0';
	}
	for(int i=0;i<10;i++){
		u32 fp1 = open(fname[i],G_WRITE);
		write(input+i,24+i,fp1);
	}
	gsys(LS_S);
	for(int i=0;i<5;i++){
		gsys(RM,fname[i]);
	}
	gsys(LS_D);

//---------------------------------Case3------------------------------

/*	char fname2[1018][20];
	int p=0;
	for(int k=2;k<15;k++){
		for(int i=50;i<=126;i++,p++){
			fname2[p][0] = i;
			for(int j=1;j<k;j++)
				fname2[p][j] = 64+j;
			fname2[p][k]='\0';
		}
	}
	for(int i=0;i<1001;i++){
		fp=open(fname2[i],G_WRITE);
		write(input+i,24+i,fp);
	}
	gsys(LS_S);

	fp = open(fname2[1000],G_READ);
	read(output+1000,1024,fp);
	char fname3[17][3];
	for(int i=0;i<17;i++){
		fname3[i][0] = 97+i;
		fname3[i][1] = 97+i;
		fname3[i][2] = '\0';
		fp = open(fname3[i],G_WRITE);
		write(input+1024*i,1024,fp);
	}
	fp=open("EA\0",G_WRITE);
	write(input+1024*100, 1024, fp);
	gsys(LS_S);*/

	//####kernel end####
}

/************************************************************************************
 *
 * Main function
 *
 ************************************************************************************/
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
	
	cudaSetDevice(2);
	mykernel<<<1,1>>>(input, output);

	cudaDeviceSynchronize();
	write_binaryFile(OUTFILE, output, FILE_STORAGE_SIZE);
	cudaDeviceReset();
	
	return 0;
}

/************************************************************************************
 *
 * The function dumps FCB information, debug uses
 *
 ************************************************************************************/
__device__ void print_FCB(const char *filename)
{
	u32 fcb=INVALID_VALUE;

	for(int i=0;i<MAX_FILE_NUM;i++){
		if(cmp_filename(fcb_table[i].name, filename)==0){
			fcb=i;
			break;
		}
	}
	printf("==========File Info Start==========\n");
	printf("file name = %s\n",fcb_table[fcb].name);
	printf("valid = %s\n",(fcb_table[fcb].valid_entry==TRUE)?"TRUE":"FALSE");
	printf("mode = %s\n",(fcb_table[fcb].mode==G_WRITE)?"WRITE":"READ");
	printf("create time = %d\n",fcb_table[fcb].create_time);
	printf("modified time = %d\n",fcb_table[fcb].modified_time);
	printf("file block num = %d\n",fcb_table[fcb].block_num);
	printf("file size = %d\n",fcb_table[fcb].file_size);
	printf("==========File Info End==========\n");
}
__device__ void print_FCB(int block)
{
	u32 fcb=block;
	printf("==========File Info Start==========\n");
	printf("file name = %s\n",fcb_table[fcb].name);
	printf("valid = %s\n",(fcb_table[fcb].valid_entry==TRUE)?"TRUE":"FALSE");
	printf("mode = %s\n",(fcb_table[fcb].mode==G_WRITE)?"WRITE":"READ");
	printf("create time = %d\n",fcb_table[fcb].create_time);
	printf("modified time = %d\n",fcb_table[fcb].modified_time);
	printf("file block num = %d\n",fcb_table[fcb].block_num);
	printf("file size = %d\n",fcb_table[fcb].file_size);
	printf("==========File Info End==========\n");
}
