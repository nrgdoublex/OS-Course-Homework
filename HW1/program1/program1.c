#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <signal.h>

void print_chld_stopped_signal(int status){
	printf("child process ");
	switch(status){
		case SIGSTOP:
			printf("get SIGSTOP signal\n");
			printf("child process stopped \n");
			break;
		case SIGTSTP:
			printf("get SIGTSTP signal\n");
			printf("child process stopped typed at terminal\n");
			break;
		case SIGTTIN:
			printf("get SIGTTIN signal\n");
			printf("child process needs terminal input\n");
			break;
		case SIGTTOU:
			printf("get SIGTTOU signal\n");
			printf("child process needs terminal output\n");
			break;
	}
}

void print_chld_signal(int status){

	printf("child process ");
	switch(status){
		case SIGHUP:
			printf("get SIGHUP signal\n");
			printf("child process is hung up\n");
			break;
		case SIGINT:
			printf("get SIGINT signal\n");
			printf("child process is interrupted\n");
			break;
		case SIGQUIT:
			printf("get SIGQUIT signal\n");
			printf("child is quit from keyboard\n");
			break;
		case SIGILL:
			printf("get SIGILL signal\n");
			printf("child process has illegal instructions\n");
			break;
		case SIGABRT:
			printf("get SIGABRT signal\n");	
			printf("child process is abort by abort signal\n");
			break;
		case SIGFPE:
			printf("get SIGFPE signal\n");	
			printf("child process has floating point exception\n");
			break;
		case SIGKILL:
			printf("get SIGKILL signal\n");	
			printf("child process is killed by kill signal\n");
			break;
		case SIGSEGV:
			printf("get SIGSEGV signal\n");	
			printf("child process has segmentation faults\n");
			break;
		case SIGPIPE:
			printf("get SIGPIPE signal\n");	
			printf("child process write pipe with no readers\n");
			break;
		case SIGALRM:
			printf("get SIGALRM signal\n");	
			printf("child process receive timer signal\n");
			break;
		case SIGTERM:
			printf("get SIGTERM signal\n");	
			printf("child process is terminated by termination signal\n");
			break;
		case SIGUSR1:
			printf("get SIGUSR1 signal\n");	
			printf("child process receive user-defined signal 1\n");
			break;
		case SIGUSR2:
			printf("get SIGUSR2 signal\n");	
			printf("child process receive user-defined signal 2\n");
			break;
		case SIGBUS:
			printf("get SIGBUS signal\n");	
			printf("child process has bus error\n");
			break;
		case SIGIO:
			printf("get SIGIO/SIGPOLL signal\n");	
			printf("I/O is available for child process\n");
			break;
		case SIGPROF:
			printf("get SIGPROF signal\n");	
			printf("profile timer expired\n");
			break;
		case SIGSYS:
			printf("get SIGSYS signal\n");	
			printf("child process system call failed\n");
			break;
		case SIGTRAP:
			printf("get SIGTRAP signal\n");	
			printf("child process reach a breakpoint\n");
			break;
		case SIGVTALRM:
			printf("get SIGVTALRM signal\n");	
			printf("virtual alarm expired\n");
			break;
		case SIGXCPU:
			printf("get SIGXCPU signal\n");	
			printf("child process cpu time limit exceeded\n");
			break;
		case SIGXFSZ:
			printf("get SIGXFSZ signal\n");	
			printf("child process file size limit exceeded\n");
			break;
		case SIGSTKFLT:
			printf("get SIGSTKFLT signal\n");	
			printf("child process has stack fault on coprocessor\n");
			break;
		case SIGPWR:
			printf("get SIGPWR signal\n");	
			printf("System power down\n");
			break;
		case SIGCHLD:
			printf("get SIGCHLD signal\n");	
			printf("child process get signal from its child\n");
			break;
		case SIGWINCH:
			printf("get SIGWINCH signal\n");	
			printf("window change size\n");
			break;
		case SIGURG:
			printf("get SIGURG signal\n");	
			printf("child process has urgent condition on sockets\n");
			break;
		default:
			printf("get signal, signal number = %d\n",status);
	}
}

int main(int argc, char *argv[]){

	if(argc<2){
		printf("please give the program name and its arguments\n");
		exit(EXIT_SUCCESS);
	}
	
	pid_t cpid;

	printf("Process start to fork\n");
	cpid=fork();
	if(cpid==-1){
		perror("fork");
		exit(EXIT_FAILURE);
	}
	
	//Child process
	if(cpid==0){
		printf("I'm the child process, my pid = %d\n",(int)getpid());
	
		int i;
		char *arg[argc];
		for(i=0;i<argc-1;i++){
			arg[i]=argv[i+1];
		}
		arg[argc-1]=NULL;

		printf("Child process start to execute the program\n");
		//printf("------------CHILD PROCESS BEGIN------------\n");
		execve(arg[0],arg,NULL);
		perror("execve");
		exit(EXIT_FAILURE);
	}
	//parent process
	else{
		int status;
		pid_t wpid;

		printf("I'm the parent process, my pid = %d\n",(int)getpid());
		do{
			wpid=waitpid(cpid,&status,WUNTRACED|WCONTINUED);
			if(wpid==-1){
				perror("wait");
				exit(EXIT_FAILURE);
			}

			printf("Parent process receiving the SIGCHLD signal\n");
			if(WIFEXITED(status)){
				printf("Normal termination with EXIT STATUS = %d\n",WEXITSTATUS(status));
			}
			else if(WIFSIGNALED(status)){
				print_chld_signal(WTERMSIG(status));
				printf("CHILD EXECUTION FAILED!!\n");
			}
			else if(WIFSTOPPED(status)){
				print_chld_stopped_signal(WSTOPSIG(status));
				printf("CHILD PROCESS STOPPED\n");
			}
			else{
				printf("CHILD PROCESS CONTINUED\n");
			}
		}while(!WIFEXITED(status)&&!WIFSIGNALED(status));
		exit(EXIT_SUCCESS);
	}
	
}
