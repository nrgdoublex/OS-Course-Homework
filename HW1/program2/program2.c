#include <linux/module.h>
#include <linux/sched.h>
#include <linux/pid.h>
#include <linux/kthread.h>
#include <linux/kernel.h>
#include <linux/err.h>
#include <linux/slab.h>
#include <linux/printk.h>
#include <linux/jiffies.h>
#include <linux/kmod.h>
#include <linux/fs.h>

MODULE_LICENSE("GPL");


struct wait_opts{
	enum pid_type wo_type;
	int wo_flags;
	struct pid *wo_pid;
	struct siginfo __user *wo_info;
	int __user *wo_stat;
	struct rusage __user *wo_rusage;
	wait_queue_t child_wait;
	int notask_error;
};

//extern struct wait_opts wo_kira;
extern long do_wait(struct wait_opts *wo);

static struct task_struct *task;

void print_chld_signal(int status){

        printk("[program2] : child process ");
        switch(status){
                case SIGHUP:
                        printk("get SIGHUP signal\n");
                        printk("[program2] : child process is hung up\n");
                        break;
                case SIGINT:
                        printk("get SIGINT signal\n");
                        printk("[program2] : child process is interrupted\n");
                        break;
                case SIGQUIT:
                        printk("get SIGQUIT signal\n");
                        printk("[program2] : child is quit from keyboard\n");
                        break;
                case SIGILL:
                        printk("get SIGILL signal\n");
                        printk("[program2] : child process has illegal instructions\n");
                        break;
                case SIGABRT:
                        printk("get SIGABRT signal\n");
                        printk("[program2] : child process is abort by abort signal\n");
                        break;
                case SIGFPE:
                        printk("get SIGFPE signal\n");
                        printk("[program2] : child process has floating point exception\n");
                        break;
                case SIGKILL:
                        printk("get SIGKILL signal\n");
                        printk("[program2] : child process is killed by kill signal\n");
                        break;
                case SIGSEGV:
                        printk("get SIGSEGV signal\n");
                        printk("[program2] : child process has segmentation faults\n");
                        break;
                case SIGPIPE:
                        printk("get SIGPIPE signal\n");
                        printk("[program2] : child process write pipe with no readers\n");
                        break;
                case SIGALRM:
                        printk("get SIGALRM signal\n");
                        printk("[program2] : child process receive timer signal\n");
                        break;
                case SIGTERM:
                        printk("get SIGTERM signal\n");
                        printk("[program2] : child process is terminated by termination signal\n");
                        break;
                case SIGUSR1:
                        printk("get SIGUSR1 signal\n");
                        printk("[program2] : child process receive user-defined signal 1\n");
                        break;
                case SIGUSR2:
                        printk("get SIGUSR2 signal\n");
                        printk("[program2] : child process receive user-defined signal 2\n");
                        break;
                case SIGBUS:
                        printk("get SIGBUS signal\n");
                        printk("[program2] : child process has bus error\n");
                        break;
                case SIGIO:
                        printk("get SIGIO/SIGPOLL signal\n");
			printk("[program2] : I/O is available for child process\n");
                        break;
                case SIGPROF:
                        printk("get SIGPROF signal\n");
                        printk("[program2] : profile timer expired\n");
                        break;
                case SIGSYS:
                        printk("get SIGSYS signal\n");
                        printk("[program2] : child process system call failed\n");
                        break;
                case SIGTRAP:
                        printk("get SIGTRAP signal\n");
                        printk("[program2] : child process reach a breakpoint\n");
                        break;
                case SIGVTALRM:
                        printk("get SIGVTALRM signal\n");
                        printk("[program2] : virtual alarm expired\n");
                        break;
                case SIGXCPU:
                        printk("get SIGXCPU signal\n");
                        printk("[program2] : child process cpu time limit exceeded\n");
                        break;
                case SIGXFSZ:
                        printk("get SIGXFSZ signal\n");
                        printk("[program2] : child process file size limit exceeded\n");
                        break;
                case SIGSTKFLT:
                        printk("get SIGSTKFLT signal\n");
                        printk("[program2] : child process has stack fault on coprocessor\n");
                        break;
                case SIGPWR:
                        printk("get SIGPWR signal\n");
                        printk("[program2] : System power down\n");
                        break;
                case SIGCHLD:
                        printk("get SIGCHLD signal\n");
                        printk("[program2] : child process get signal from its child\n");
                        break;
                case SIGWINCH:
                        printk("get SIGWINCH signal\n");
                        printk("[program2] : window change size\n");
                        break;
                case SIGURG:
                        printk("get SIGURG signal\n");
                        printk("[program2] : child process has urgent condition on sockets\n");
                        break;
                default:
                        printk("get signal, signal number = %d\n",status);
        }
}


//implement exec function
int my_exec(void){
	int result;
	const char path[]="/home/kira/work/os_hw1/program2/test";
	const char *const argv[]={path,NULL,NULL};
	const char *const envp[]={"HOME=/","PATH=/sbin:/user/sbin:/bin:/usr/bin",NULL};
	//result=call_usermodehelper(path,argv,envp,UMH_WAIT_PROC);

	result=do_execve(path,argv,envp);

	//if exec success
	if(!result)
		return 0;
	//ifexec failed
	do_exit(result);
}

//implement wait function
void my_wait(pid_t pid){

	long status;
	int retval;	
	struct wait_opts wo;
//	struct siginfo sig_info;
	struct pid *wo_pid=NULL;
	enum pid_type type;
	type=PIDTYPE_PID;
	wo_pid=find_get_pid(pid);
	retval=-1;

	wo.wo_type=type;
	wo.wo_pid=wo_pid;
	wo.wo_flags=WEXITED;
	wo.wo_info=NULL;
	wo.wo_stat=(int __user *)&retval;
	wo.wo_rusage=NULL;
	status=do_wait(&wo);


	// output child process exit status
	//printk("return value = %d\n",retval);
	if(retval & 0x80){
		printk("[program2] : The child process is core-dumped\n");
		printk("[program2] : The return signal number = %d\n",retval&0x7f);
		print_chld_signal(retval&0x7f);
	}
	else if(retval & 0x7f){
		printk("[program2] : The child process is terminated\n");
		printk("[program2] : The return signal number = %d\n",retval&0x7f);
		print_chld_signal(retval&0x7f);
	}
	else{
		printk("[program2] : The child process exited normally\n");
		printk("The exit status = %d\n",retval>>8);
	}
	put_pid(wo_pid);
	return;
}

//implement fork function
int my_fork(void *argc){
	int i;
	long pid=0;
	struct k_sigaction *k_action = &current->sighand->action[0];

	//set default sigaction for current process
	for(i=0;i<_NSIG;i++){
		k_action->sa.sa_handler = SIG_DFL;
		k_action->sa.sa_flags = 0;
		k_action->sa.sa_restorer = NULL;
		sigemptyset(&k_action->sa.sa_mask);
		k_action++;
	}
	//fork process
	pid=do_fork(SIGCHLD,(unsigned long)&my_exec,0,NULL,NULL);


	printk("[program2] : The child process has pid = %ld\n",pid);
	printk("[program2] : This is the parent process,pid = %d\n",(int)current->pid);

	my_wait(pid);
	return 0;
}

static int __init program2_init(void){
	char program2[9]="program2";

	printk("[program2] : module_init\n");
	printk("[program2] : module_init create kthread start\n");
	//create a kthread
	task=kthread_create(&my_fork,NULL,program2);
	//wake up new thread if ok
	if(!IS_ERR(task)){
		printk("[program2] : module_init kthread start\n");
		wake_up_process(task);
	}
	return 0;
}

static void __exit program2_exit(void){
	printk("[program2] : module_exit\n");
}

module_init(program2_init);
module_exit(program2_exit);
