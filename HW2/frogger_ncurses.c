#include <ncurses.h>
#include <pthread.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>


#define NUM_LOG_LINE 5 
#define NUM_LOGS_PER_LINE 8
#define ROW_END 0
#define ROW_SCRATCH (NUM_LOG_LINE + ROW_END + 1)

typedef enum{LEFT,RIGHT,UP,DOWN} Direction;


typedef enum{PLAYING,WIN,LOSE,QUIT} State;
static State gamestate;

typedef struct{
	int row;
	int col;
	char under_char;
}frog_info;

static frog_info frog;

typedef struct{
	int id;
	Direction dir;
	int row;
	int sleeptime;
}log_line_info;

static log_line_info logs_info[NUM_LOG_LINE];

pthread_t log_thread[NUM_LOG_LINE];
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

const char grass[]="||||||||";
const char logs[]="====    ";
const char frog_entity[] = "O";
#define LINE_LENGTH ((sizeof(logs)-1)*NUM_LOGS_PER_LINE+1)
char lines[NUM_LOG_LINE][LINE_LENGTH];


void init_game(int num_logs_per_line){
	int i,j;
	

	for(i=0;i<num_logs_per_line;i++){
		strcat(lines[ROW_END],grass);
		strcat(lines[ROW_SCRATCH],grass);
		for(j=ROW_END+1;j<ROW_SCRATCH;j++)
			strcat(lines[j],logs);
	}

	for(i=0;i<num_logs_per_line;i++){
		mvprintw(ROW_END,0,lines[ROW_END]);
		mvprintw(ROW_SCRATCH,0,lines[ROW_SCRATCH]);
		for(j=ROW_END+1;j<ROW_SCRATCH;j++)
			mvprintw(j,0,lines[j]);
	}
};

void init_log_line(log_line_info *log,int id,Direction dir){
	log->id=id;
	log->dir=dir;
	log->row=id+ROW_END+1;
	log->sleeptime=(rand()%10+1)*100000;
}

void init_frog(){
	frog.row=ROW_SCRATCH;
	frog.col=LINE_LENGTH/2;
	frog.under_char = '|';
	mvprintw(frog.row,frog.col,frog_entity);
}

void log_line_move(log_line_info *log){

	pthread_mutex_lock(&mutex);

	char tmpline[LINE_LENGTH];
	int length;
	int row = log->id+ROW_END+1;

	length=strlen(lines[row]);
	strcpy(tmpline,lines[row]);
	if(log->dir==LEFT){
		strncpy(lines[row],(const char *)(tmpline+1),length-1);
		lines[row][length-1]=tmpline[0];
	}
	else{
		strncpy(lines[row]+1,(const char *)tmpline,length-1);
		lines[row][0]=tmpline[length-1];
	}
	mvprintw(row,0,lines[row]);

	if(frog.row==row){
		frog.col=frog.col+((log->dir==LEFT)?-1:1);
		mvprintw(frog.row,frog.col,frog_entity);
	}

	if(frog.col<0||frog.col>(LINE_LENGTH-2))
		gamestate=LOSE;
	move(0,0);
	refresh();
		
	pthread_mutex_unlock(&mutex);
}

void *log_function(void *data){
	log_line_info *log;
	log = (log_line_info *)data;
	while(gamestate==PLAYING){
		log_line_move(log);
		usleep(log->sleeptime);
	}
	return NULL;
}

void move_frog(State state){
	pthread_mutex_lock(&mutex);

	mvprintw(frog.row,frog.col,&(frog.under_char));
	switch(state){
		case UP:
			if(frog.row>ROW_END)
				frog.row--;
			break;
		case DOWN:
			if(frog.row<ROW_SCRATCH)
				frog.row++;
			break;
		case LEFT:
			if(frog.col>0)
				frog.col--;
			break;
		case RIGHT:
			if(frog.col<LINE_LENGTH-2)
				frog.col++;
			break;
		default:
			break;
	}
	mvprintw(frog.row,frog.col,frog_entity);
	frog.under_char=lines[frog.row][frog.col];
	if(frog.under_char==' ')
		gamestate=LOSE;
	if(frog.row==ROW_END)
		gamestate=WIN;

	move(0,0);
	refresh();
	pthread_mutex_unlock(&mutex);
}

int main(int argc,char *argv[]){
	int i;
	char ch;
	Direction dir=LEFT;

	srand(time(NULL));

	initscr();
	clear();
	cbreak();
	noecho();
	keypad(stdscr,TRUE);
	timeout(100);

	init_game(NUM_LOGS_PER_LINE);
	for(i=0;i<NUM_LOG_LINE;i++){
		init_log_line(&logs_info[i],i,dir);
		dir=(dir==LEFT)?RIGHT:LEFT;
	}
	init_frog();
	gamestate=PLAYING;

	for(i=0;i<NUM_LOG_LINE;i++)
		pthread_create(&log_thread[i],NULL,&log_function,&logs_info[i]);

	while(gamestate==PLAYING){
		ch=getch();
		switch(ch){
			case 'w':
				move_frog(UP);
				break;
			case 's':
				move_frog(DOWN);
				break;
			case 'a':
				move_frog(LEFT);
				break;
			case 'd':
				move_frog(RIGHT);
				break;
			case 'q':
				gamestate=QUIT;
				break;
			default:
				break;
		}
	}

	for(i=0;i<NUM_LOG_LINE;i++)
		pthread_join(log_thread[i],NULL);

	echo();	
	nocbreak();
	endwin();

	switch(gamestate){
		case WIN:
			printf("You WIN!!\n");
			break;
		case LOSE:
			printf("You LOSE!!\n");
			break;
		case QUIT:
			printf("You QUIT!!\n");
			break;
		default:
			printf("This state should be unreached\n");
			break;
	}

	exit(EXIT_SUCCESS);
}
