#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/cdev.h>
#include <linux/fs.h>
#include <linux/errno.h>
#include <linux/types.h>
#include <linux/init.h>
#include <linux/slab.h>
#include <linux/sched.h>
#include <linux/workqueue.h>
#include <linux/atomic.h>
#include <asm/uaccess.h>
#include <linux/delay.h>
#include "ioc_hw5.h"

#define DUBUG
#define TRUE				(1)
#define FALSE				(0)

#define DMA_BUFSIZE			(64)
#define DMASTUIDADDR		(0x0)
#define DMARWOKADDR			(0x4)
#define DMAIOCOKADDR		(0x8)
#define DMAIRQOKADDR		(0xC)
#define DMACOUNTADDR		(0x10)
#define DMAANSADDR			(0x14)
#define DMAREADABLEADDR		(0x18)
#define DMABLOCKADDR		(0x1C)
#define DMAOPCODEADDR		(0x20)
#define DMAOPERANDBADDR		(0x21)
#define DMAOPERANDCADDR		(0x25)

#define DRV_NR_DEVS			(1)

#define RW_OK				(1)
#define RW_NOT_OK			(0)
#define IOC_OK				(1)
#define IOC_NOT_OK			(0)
#define IRQ_OK				(1)
#define IRQ_NOT_OK			(0)
#define BLOCKING_IO			(1)
#define NONBLOCKING_IO		(0)
#define DEV_READABLE		(1)
#define DEV_NOT_READABLE	(0)

#undef PMEG
#define PMEG(fmt, args...) printk(KERN_DEBUG "OS_HW5:" fmt, ## args)

//the data structure of arithmetic
struct dataIn{
	char a;
	int b;
	short c;
};

// the data structure of device
struct oshw5_dev{
	struct work_struct work;
	struct workqueue_struct *workqueue;
	struct semaphore sem;
	int drv_major;					//device major number
	int drv_minor;					//device minor number
	struct cdev cdev;				//char device structure
};

//global device structure
static struct oshw5_dev *drv_dev;
//dma buffer of the device
void *dma_buf;

MODULE_AUTHOR("Yu-Jun Li");
MODULE_LICENSE("GPL");


/*---------------------DMA Access Function---------------------*/
void myoutb(unsigned char data, unsigned short int port)
{
	unsigned char *value = (unsigned char *)(dma_buf+port);
	*value=data;
}

void myoutw(unsigned short data, unsigned short int port)
{
	unsigned short *value = (unsigned short *)(dma_buf+port);
	*value=data;
}

void myoutl(unsigned int data, unsigned short int port)
{
	unsigned int *value = (unsigned int *)(dma_buf+port);
	*value=data;
}

unsigned char myinb(unsigned short int port)
{
	unsigned char *value = (unsigned char *)(dma_buf+port);
	return *value;
}

unsigned short myinw(unsigned short int port)
{
	unsigned short *value = (unsigned short *)(dma_buf+port);
	return *value;
}

unsigned int myinl(unsigned short int port)
{
	unsigned int *value = (unsigned int *)(dma_buf+port);
	return *value;
}

/*---------------------Arithmetic Functions---------------------*/
int prime(int base, short nth)
{
	int fnd=0;
	int i, num, isPrime;

	num = base;
	while(fnd != nth){
		isPrime=1;
		num++;
		for(i=2;i<=num/2;i++){
			if(num%i==0){
				isPrime=0;
				break;
			}
		}
		if(isPrime)
			fnd++;
	}
	return num;
}

int arithmetic(char operator, int operand1, short operand2)
{
	int ans;

	switch(operator){
		case '+':
			ans=operand1+operand2;
			break;
		case '-':
			ans=operand1-operand2;
			break;
		case '*':
			ans=operand1*operand2;
			break;
		case '/':
			ans=operand1/operand2;
			break;
		case 'p':
			ans=prime(operand1,operand2);
			break;
		default:
			ans=0;
			return -EINVAL;
	}
	PMEG("%s(): %d %c %d = %d\n",__FUNCTION__,operand1, operator, operand2, ans);
	return ans;
}

void arithmetic_work(struct work_struct *work)
{
	//calculate arithmetics
	myoutl(arithmetic(myinb(DMAOPCODEADDR),myinl(DMAOPERANDBADDR),myinw(DMAOPERANDCADDR)),DMAANSADDR);

	//set readable register 	
	myoutl(DEV_READABLE, DMAREADABLEADDR);
}

/*---------------------File Operations---------------------*/
int drv_open(struct inode *inode, struct file *filp)
{
	struct oshw5_dev *dev;
	dev=container_of(inode->i_cdev, struct oshw5_dev, cdev);
	filp->private_data=dev;

	PMEG("%s():device open\n",__FUNCTION__);
	
	return 0;
}

int drv_release(struct inode *inode, struct file *filp)
{
	PMEG("%s():device close\n",__FUNCTION__);
	return 0;
}

ssize_t drv_read(struct file *filp, char __user *buf, size_t count, loff_t *f_ops)
{

	int retval = 0,value;

	if(down_interruptible(&drv_dev->sem))
		return -ERESTARTSYS;

	if(myinl(DMAREADABLEADDR)==DEV_NOT_READABLE){
		retval = -EBUSY;
		goto out;
	}

	value = myinl(DMAANSADDR);
	
	PMEG("%s(): ans = %d\n",__FUNCTION__,value);
	if(copy_to_user(buf,(const void *)&value,sizeof(int))){
		retval = -EFAULT;
		goto out;
	}

	myoutl(0,DMAANSADDR);

out:
	up(&drv_dev->sem);
	return retval;
}

ssize_t drv_write(struct file *filp, const char __user *buf, size_t count, loff_t *f_ops)
{
	int retval = 0,value;
	struct dataIn input;

	if(down_interruptible(&drv_dev->sem))
		return -ERESTARTSYS;

	if(copy_from_user(&input,buf,sizeof(struct dataIn))){
		retval = -EFAULT;
		goto out;
	}

	myoutl(DEV_NOT_READABLE,DMAREADABLEADDR);					//disable read
	myoutb(input.a,DMAOPCODEADDR);								//write opcode
	myoutl(input.b,DMAOPERANDBADDR);							//write operand1
	myoutw(input.c,DMAOPERANDCADDR);							//write operand2

	PMEG("%s(): queue work\n",__FUNCTION__);
//	retval = schedule_work(&drv_dev->work);
	retval = queue_work(drv_dev->workqueue, &drv_dev->work);	//send work to workqueue

	value=myinl(DMABLOCKADDR);
	if(value==NONBLOCKING_IO){									//if blocking IO is enabled
		PMEG("%s(): non-block\n",__FUNCTION__);
	}
	else if(value==BLOCKING_IO){								//if non-blocking IO is enabled
		while(myinl(DMAREADABLEADDR)==DEV_NOT_READABLE){
			msleep(1);
		}
		PMEG("%s(): block\n",__FUNCTION__);
	}
	else
		PMEG("%s(): BLOCK Register Error\n",__FUNCTION__);

out:
	up(&drv_dev->sem);
	return retval;
}

long drv_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
	int err= 0;
	unsigned int value;
	unsigned int readable;

	if(_IOC_TYPE(cmd) != HW5_IOC_MAGIC) return -ENOTTY;
	if(_IOC_NR(cmd) > HW5_IOC_MAXNR) return -ENOTTY;

	if(_IOC_DIR(cmd) & _IOC_READ)
		err = !access_ok(VERIFY_WRITE, (void __user *)arg, _IOC_SIZE(cmd));
	else if(_IOC_DIR(cmd) & _IOC_WRITE)
		err = !access_ok(VERIFY_READ, (void __user *)arg, _IOC_SIZE(cmd));
	if(err) return -EFAULT;

	switch(cmd){
		case HW5_IOCSETSTUID:
			value = *(unsigned int *)arg;
			myoutl(value,DMASTUIDADDR);
			if(*(unsigned int *)arg==myinl(DMASTUIDADDR))
				PMEG("%s(): My STUID is = %d\n",__FUNCTION__,myinl(DMASTUIDADDR));
			else
				PMEG("%s(): Set STUID failed\n",__FUNCTION__);
			break;
		case HW5_IOCSETRWOK:
			value = *(unsigned int *)arg;
			if(value != RW_OK && value != RW_NOT_OK){
				PMEG("%s(): Invalid value for RW Register\n",__FUNCTION__);
				return -EINVAL;
			}
			myoutl(value,DMARWOKADDR);
			if(*(unsigned int *)arg==myinl(DMARWOKADDR))
				PMEG("%s(): RW OK\n",__FUNCTION__);
			else
				PMEG("%s(): RW failed\n",__FUNCTION__);
			break;
		case HW5_IOCSETIOCOK:
			value = *(unsigned int *)arg;
			if(value != IOC_OK && value != IOC_NOT_OK){
				PMEG("%s(): Invalid value for IOC Register\n",__FUNCTION__);
				return -EINVAL;
			}
			myoutl(value,DMAIOCOKADDR);
			if(*(unsigned int *)arg==myinl(DMAIOCOKADDR))
				PMEG("%s(): IOC OK\n",__FUNCTION__);
			else
				PMEG("%s(): IOC failed\n",__FUNCTION__);
			break;
		case HW5_IOCSETIRQOK:
			value = *(unsigned int *)arg;
			if(value != IRQ_OK && value != IRQ_NOT_OK){
				PMEG("%s(): Invalid value for IRQ Register\n",__FUNCTION__);
				return -EINVAL;
			}
			myoutl(value,DMAIRQOKADDR);
			if(*(unsigned int *)arg==myinl(DMAIRQOKADDR))
				PMEG("%s(): IRQ OK\n",__FUNCTION__);
			else
				PMEG("%s(): IRQ failed\n",__FUNCTION__);
			break;
		case HW5_IOCSETBLOCK:
			value = *(unsigned int *)arg;
			if(value != BLOCKING_IO && value != NONBLOCKING_IO){
				PMEG("%s(): Invalid value for BLOCKING Register\n",__FUNCTION__);
				return -EINVAL;
			}
			myoutl(value,DMABLOCKADDR);
			if(*(unsigned int *)arg==myinl(DMABLOCKADDR))
				if(myinl(DMABLOCKADDR)==BLOCKING_IO)
					PMEG("%s(): Blocking IO\n",__FUNCTION__);
				else
					PMEG("%s(): Non-Blocking IO\n",__FUNCTION__);
			else
				PMEG("%s(): Set STUID failed\n",__FUNCTION__);
			break;
		case HW5_IOCWAITREADABLE:
			while((readable=myinl(DMAREADABLEADDR))==DEV_NOT_READABLE){
				msleep(1);
			}
			*(unsigned int *)arg = readable;
			PMEG("%s(): wait readable = %d\n",__FUNCTION__,readable);
			break;
		default:
			return -ENOTTY;
	}

	return 0;
}

/*---------------------DMA Buffer Alloc/Free---------------------*/
int init_dma_buf(void)
{
	dma_buf = kmalloc(DMA_BUFSIZE,GFP_KERNEL);
	memset(dma_buf,0,DMA_BUFSIZE);
	if(!dma_buf){
		dma_buf=NULL;
		return -ENOMEM;
	}
	else
		return 0;

}

void free_dma_buf(void)
{
	if(dma_buf){
		kfree(dma_buf);
		dma_buf=NULL;
	}
}

struct file_operations oshw5_fops = {
	.owner			=THIS_MODULE,
	.read			=drv_read,
	.write			=drv_write,
	.unlocked_ioctl	=drv_ioctl,
	.open			=drv_open,
	.release		=drv_release,
};

int drv_setup_cdev(struct oshw5_dev *dev)
{
	//initialize this device
	int err, devno = MKDEV(dev->drv_major,dev->drv_minor);

	cdev_init(&dev->cdev,&oshw5_fops);
	dev->cdev.owner	= THIS_MODULE;
	dev->cdev.ops	= &oshw5_fops;
	err=cdev_add(&dev->cdev, devno, DRV_NR_DEVS);

	return err;
}

/*---------------------Module Functions---------------------*/
int __init init_modules(void)
{

	int result;
	dev_t devno;

	PMEG("%s():--------------------START--------------------\n",__FUNCTION__);

	//register device
	if((result = alloc_chrdev_region(&devno, 0, DRV_NR_DEVS, "os_hw5"))){						//register char device
		PMEG("%s():can't get device number\n",__FUNCTION__);
		goto register_fail;
	}

	//allocate device structure
	drv_dev = (struct oshw5_dev *) kmalloc(DRV_NR_DEVS * sizeof(struct oshw5_dev),GFP_KERNEL);	//allocate driver data structure
	if(!drv_dev){
		result = -ENOMEM;
		PMEG("%s():Error %d when allocating device\n",__FUNCTION__,result);
		goto allocate_fail;
	}
	memset(drv_dev,0,DRV_NR_DEVS * sizeof(struct oshw5_dev));

	//setup driver structure
	drv_dev->drv_major = MAJOR(devno);															//get device major number
	drv_dev->drv_minor = MINOR(devno);															//get device minor number
	PMEG("%s():register chrdev(%d,%d)\n",__FUNCTION__,drv_dev->drv_major,drv_dev->drv_minor);
	sema_init(&drv_dev->sem,1);																	//initialize semaphore
	if((result=drv_setup_cdev(drv_dev))){														//setup device structure	
		PMEG("%s():Error %d when adding device\n",__FUNCTION__,result);
		goto cdev_fail;
	}
	//we don't initialize work struct here
	INIT_WORK(&drv_dev->work, arithmetic_work);
	drv_dev->workqueue = create_workqueue("oshw5_wq");											//create workqueue


	//initialize dma buffer
	if((result=init_dma_buf())){
		PMEG("%s:allocating dma buffer failed\n",__FUNCTION__);
		goto dma_buf_fail;
	}
	PMEG("%s():allocate dma buffer\n",__FUNCTION__);


	return 0;

dma_buf_fail:
	destroy_workqueue(drv_dev->workqueue);
	cdev_del(&drv_dev->cdev);
cdev_fail:
	kfree(&drv_dev);
allocate_fail:
	unregister_chrdev_region(devno,DRV_NR_DEVS);
register_fail:

	return result;
}

void __exit exit_modules(void)
{
	dev_t devno = MKDEV(drv_dev->drv_major, drv_dev->drv_minor);

	free_dma_buf();										//free dma buffer
	PMEG("%s():free dma buffer\n",__FUNCTION__);

	destroy_workqueue(drv_dev->workqueue);				//destroy workqueue
	if(drv_dev){
		cdev_del(&drv_dev->cdev);						//delete char device
		kfree(drv_dev);									//free the driver data structure
		drv_dev=NULL;									//set the pointer null
	}

	unregister_chrdev_region(devno, DRV_NR_DEVS);		//unregister char device
	PMEG("%s():unregister chrdev\n",__FUNCTION__);
	PMEG("%s():--------------------END--------------------\n",__FUNCTION__);
	return;
}

module_init(init_modules);
module_exit(exit_modules);
