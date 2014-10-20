#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <wait.h>
#include <unistd.h>

#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>

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


int main(int argc,char *argv[]){

	if(argc<2){
		printf("please give the program name\n");
		exit(EXIT_SUCCESS);
	}

	if(argc>=2){
		int i;
		int fd;
		void* file_mem;
		const int file_len=argc*100;
		int len_write;
		len_write=0;

		//mmap
		fd = open("./tmp.txt",O_RDWR|O_CREAT,S_IRUSR|S_IWUSR);
		if(fd==-1){
			perror("mmap");
			exit(EXIT_FAILURE);
		}
		lseek(fd,file_len+1,SEEK_SET);
		write(fd,"",1);
		lseek(fd,0,SEEK_SET);
		file_mem=mmap(0,file_len,PROT_WRITE|PROT_READ,MAP_SHARED,fd,0);
		close(fd);
		
		
		pid_t mpid;
		mpid=fork();
		if(mpid==-1){
			perror("fork");
			exit(EXIT_FAILURE);
		}
		//execution process
		if(mpid==0){
			for(i=1;i<argc;i++){
				pid_t cpid,w;
				cpid=fork();
				if(cpid==-1){
					perror("fork");
					exit(EXIT_FAILURE);
				}
				if(cpid==0){
					int fd;
					continue;
				}
				else{
					// for wait
					int status;
					pid_t pid;
					char *arg[2]={argv[i],NULL};
					pid = getpid();
					*(volatile int*)(file_mem+sizeof(int)*(3*i-3))=(int)pid;
					
					w=waitpid(cpid,&status,WUNTRACED|WCONTINUED);
					if(w==-1){
						perror("wait");
						exit(EXIT_FAILURE);
					}
					if(WIFEXITED(status)){
						*(volatile int*)(file_mem+sizeof(int)*(3*i-2))=0x1;
						*(volatile int*)(file_mem+sizeof(int)*(3*i-1))=WEXITSTATUS(status);
					}
					else if(WIFSIGNALED(status)){
						*(volatile int*)(file_mem+sizeof(int)*(3*i-2))=0x2;
						*(volatile int*)(file_mem+sizeof(int)*(3*i-1))=WTERMSIG(status);
					}
					else if(WIFSTOPPED(status)){
						*(volatile int*)(file_mem+sizeof(int)*(3*i-2))=0x3;
						*(volatile int*)(file_mem+sizeof(int)*(3*i-1))=WSTOPSIG(status);
					}
					else if(WIFCONTINUED(status)){
						*(volatile int*)(file_mem+sizeof(int)*(3*i-2))=0x4;
						*(volatile int*)(file_mem+sizeof(int)*(3*i-1))=0;
					}
					munmap(file_mem,file_len);
	
					execve(arg[0],arg,NULL);
					printf("execve failed, pid = %d\n",(int)pid);
					exit(EXIT_FAILURE);
				}
			}
		}
		// monitor process
		else{
			int pid[argc],wait_status[argc],exit_status[argc];
			pid_t monitor;
			int m_status,idx;
			monitor=wait(&m_status);
			if(monitor==-1){
				perror("wait in monitor process");
				exit(EXIT_FAILURE);
			}
		
			//get first child's state
			pid[0]=getpid();
			if(WIFEXITED(m_status)){
				wait_status[0]=1;
				exit_status[0]=WEXITSTATUS(m_status);
			}
			else if(WIFSIGNALED(m_status)){
				wait_status[0]=2;
				exit_status[0]=WTERMSIG(m_status);
			}
			if(WIFSTOPPED(m_status)){
				wait_status[0]=3;
				exit_status[0]=WSTOPSIG(m_status);
			}
			if(WIFCONTINUED(m_status)){
				wait_status[0]=4;
				exit_status[0]=0;
			}
			//get descendant's states
			for(idx=1;idx<argc;idx++){
				pid[idx]=*(int*)(file_mem+0x00);
				wait_status[idx]=*(int*)(file_mem+sizeof(int));
				exit_status[idx]=*(int*)(file_mem+sizeof(int)*2);
				file_mem+=sizeof(int)*3;
			}
			munmap(file_mem,file_len);

			//print process tree
			printf("the process tree : ");
			for(idx=0;idx<argc;idx++){
				printf("%d",pid[idx]);
				if(idx!=argc-1)
					printf("->");
			}
			printf("\n");

			//print child process status
			for(idx=argc-1;idx>=0;idx--){
				if(idx!=argc-1){
					printf("The child process(pid=%d) of parent process(pid=%d) ",pid[idx+1],pid[idx]);
					switch(wait_status[idx]){
						case 1:
							printf("has normal execution\n");
							printf("Its exit status = %d\n\n",exit_status[idx]);
							break;
						case 2:
							printf("is terminated by signal\n");
							printf("Its signal number = %d\n",exit_status[idx]);
							print_chld_signal(exit_status[idx]);
							printf("\n");
							break;
						case 3:
							printf("is stopped by signal\n");
							printf("Its signal number = %d\n\n",exit_status[idx]);
							print_chld_stopped_signal(exit_status[idx]);
							printf("\n");
							break;
						case 4:
							printf("resume its execution\n\n");
							break;
						default:
							printf("\ninvalid mmem accessed\n\n");
					}
				}
			}
			printf("\nMyfork process(pid=%d) execute normally\n",pid[0]);
		}
	}
	return 0;
}
