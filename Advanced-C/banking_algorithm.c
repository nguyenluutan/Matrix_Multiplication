#include <stdio.h>

struct process
{
    int max[10],allocate[10],need[10],nallocate[10],nneed[10];
    int finish;
} p[10];
int avail[10],navail[10],req[10];
int i,j,m,n,sum,s,pno;
int generate();
void newstatus();

int main()
{
    int k,l;
    printf("ENTER THE TOTAL NUMBER OF RESOURCES :");
    scanf("%d",&n);
    printf("ENTER THE NUMBER OF AVAILABLE UNITS IN EACH RESOURCE TYPE:");
    for(i=0; i<n; i++)
        scanf("%d",&avail[i]);
    printf("ENTER THE TOTAL NUMBER OF PROCESSES :");
    scanf("%d",&m);
    printf("ENTER THE MAXIMUM NUMBER OF UNITS NEEDED IN");
    printf("EACH RESOURCE TYPE FOR EACH PROCESS:");
    for(i=0; i<m; i++)
        for(j=0; j<n; j++)
            scanf("%d",&p[i].max[j]);
    printf("ENTER THE ALLOCATED NUMBER OF UNITS IN");
    printf("EACH RESOURCE TYPE FOR EACH PROCESS:");
    for(i=0; i<m; i++)
        for(j=0; j<n; j++)
        {
            scanf("%d",&p[i].allocate[j]);
            p[i].need[j]=p[i].max[j]-p[i].allocate[j];
        }
    for(j=0; j<n; j++)
    {
        sum=0;
        for(i=0; i<m; i++)
            sum+=p[i].allocate[j];
        avail[j]=avail[j]-sum;
    }
    newstatus();
    printf("THE NEED MATRIX IS:");
    for(i=0; i<n; i++)
        printf("R%d ",i+1);
    for(i=0; i<m; i++)
    {
        printf("");
        for(j=0; j<n; j++)
            printf("%d  ",p[i].need[j]);
    }
    l=generate();
    if(l==0)
        printf("THE SYSTEM IS NOT IN THE SAFE STATE!!!");
    else
    {
        newstatus();
        printf("ENTER THE PROCESS NUMBER AND ITS REQUEST:");
        scanf("%d",&pno);
        pno=pno-1;
        k=s=0;
        for(i=0; i<n; i++)
        {
            scanf("%d",&req[i]);
            if((req[i]>avail[i])||(req[i]>p[pno].need[i]))
                k=1;
            if(req[i]!=p[pno].need[i])
                s=1;
        }
        if(k==1)
            printf("THE REQUEST CANNOT BE GRANTED!!!");
        else if(k==0)
        {
            for(i=0; i<n; i++)
            {
                p[pno].nneed[i]=(p[pno].nneed[i])-req[i];
                p[pno].nallocate[i]=(p[pno].nallocate[i])+req[i];
                navail[i]=navail[i]-req[i];
                if(s==0)
                    navail[i]=navail[i]+p[pno].nallocate[i];
            }
            printf("");
            if(s==0)
            {
                printf("P%d ",pno+1);
                p[pno-1].finish=1;
            }
            l=generate();
            if(l==0)
                printf("THE REQUEST CANNOT BE GRANTED!!!");
            if(l==1)
                printf("THE REQUEST IS GRANTED!!!");
        }
    }

    return 0;
}
void newstatus()
{
    for(i=0; i<n; i++)
        navail[i]=avail[i];
    for(i=0; i<m; i++)
    {
        for(j=0; j<n; j++)
        {
            p[i].nallocate[j]=p[i].allocate[j];
            p[i].nneed[j]=p[i].need[j];
        }
        p[i].finish=0;
    }
}
int generate()
{
    int l,q,k,f;
    l=0;
    while(1)
    {
        for(i=0; i<m; i++)
        {
            if(p[i].finish==0)
            {
                k=1;
                for(j=0; j<n; j++)
                    if(p[i].nneed[j]>navail[j])
                        k=0;
                if(k==0)
                    l++;
                else if(k==1)
                {
                    printf("P%d ",i+1);
                    for(j=0; j<n; j++)
                        navail[j]+=p[i].nallocate[j];
                    p[i].finish=1;
                }
            }
        }
        if(l>m)
            return 0;
        q=0;
        for(f=0; f<m; f++)
            if(p[f].finish==1)
                q++;
        if(q==m)
        {
            printf("ABOVE IS THE SAFE SEQUENCE AND THE SYSTEM IS IN SAFE STATE!!!");
            return 1;
        }
    }
}
