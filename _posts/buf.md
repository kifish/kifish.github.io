

```c
#include<stdio.h>
#define IN 1 //inside a word
#define OUT 0 // outside a word

//count lines,words,and characters in input

int main(){
    int c,nl,nw,nc,state;
    state = OUT;
    nl = nw = nc =0;
    while((c = getchar()) != EOF){
        ++nc;
        if(c == '\n') ++nl;
        if(c == ' ' || c == '\n' || c == '\t') state = OUT;
        else if(state == OUT){
            state = IN;
            ++nw;
        }
    }
    printf("%d %d %d\n",nl,nw,nc);
    return 0;
}
```



```c
#include<stdio.h>
//count digits,white space,others
int main()
{
    int c,i,n_white,n_other;
    int n_digit[10];
    n_white = n_other = 0;
    for(int i = 0;i<10;++i) n_digit[i] = 0;
    while((c = getchar()) != EOF ){
        if(c >= '0' && c <= '9') ++n_digit[c-'0'];
        else if (c == ' ' || c == '\n' || c == '\t') ++n_white;
        else ++n_other;
    }
    printf("digits =");
    for(int i = 0;i < 10;++i) printf(" %d",n_digit[i]);
    printf(", white space = %d, other = %d\n",n_white,n_other);
    return 0;
}
```