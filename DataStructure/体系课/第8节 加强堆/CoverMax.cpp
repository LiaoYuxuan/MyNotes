#include <iostream>
#include <vector>
#include <queue>
#include <time.h>
#include <limits.h>
using namespace std;

class Line{
public:
    //成员变量
    int start;
    int end;

    void initial(int s, int e)
    {
        start = s;
        end = e;
    }
};

// 产生随机线段
vector<vector<int>> generateLines(int maxN, int minL, int maxR){
    // 产生[0,maxN-1]数量的线条
    int size = (rand()%(maxN-0))+0;
    // c++中使用vector创建二维数组的方法：vector<vector<int>>b(rows,vector<int>(col));
    vector<vector<int>> lines(size,vector<int>(2));
    for (int i = 0; i < size; i++)
    {
        int s = (rand()%(maxR-minL+1))+minL;
        int e = (rand()%(maxR-minL+1))+minL;
        if (s==e)    e = s+1;
        lines[i][0] = min(s,e);
        lines[i][1] = max(s,e);
    }
    return lines;
}

// 暴力解法
int worseSolution(vector<vector<int>> lines){
    // 第一步：找出所有线段的最小起点，和最大终点
    int min_s = INT_MAX;
    int max_e = INT_MIN;
    for (int i = 0; i < lines.size(); i++)
    {
        min_s = min(min_s,lines[i][0]);
        max_e = max(max_e,lines[i][1]);
    }
    // 第二步：对于每个0.5位置上，统计有多少条线段，输出最大值即可
    int cover = 0;
    for (double d = min_s + 0.5; d < max_e; d+=1)
    {
        int cur = 0;
        for (int i = 0; i < lines.size(); i++)
        {
            if (d > lines[i][0] && d < lines[i][1])    cur++;
        }
        cover = max(cur,cover);
    }
    return cover;
    
}

bool comp(Line l, Line r){
    return l.start < r.start;
}

int heapSolution(vector<vector<int>> lines){
    // 第一步：把二维数组转成对象数组
    vector<Line> pre_lines(lines.size());
    for (int i = 0; i < lines.size(); i++)
    {
        Line tmp = Line();
        tmp.initial(lines[i][0],lines[i][1]);
        pre_lines[i] = tmp;
    }

    // // 测试用
    // for (int i = 0; i < pre_lines.size(); i++)
    // {
    //     cout << pre_lines[i].start << "," << pre_lines[i].end << endl;
    // }
    
    // 第二步：使用自定义的比较器对线段进行排序
    sort(pre_lines.begin(),pre_lines.end(),comp);

    // // 测试用
    // for (int i = 0; i < pre_lines.size(); i++)
    // {
    //     cout << pre_lines[i].start << "," << pre_lines[i].end << endl;
    // }

    // 第三步：遍历所有线段，使用最小堆堆
    int cover = 0;
    // 小根堆
    priority_queue<int, vector<int>, greater<int>> min_heap;
    for (int i = 0; i < pre_lines.size(); i++)
    {
        // 删除堆中小于等于当前线段起点的元素
        while (!min_heap.empty() && min_heap.top()<=pre_lines[i].start)    min_heap.pop();
        // 插入当前线段的终点
        min_heap.push(pre_lines[i].end);
        // cout << min_heap.size() << endl;
        cover = max(cover,int(min_heap.size()));
    }
    
    return cover;
}

int main(){
    int maxN = 1000;
    int minL = 0;
    int maxR = 2000;
    int testTimes = 20000;
    bool succeed = true;
    printf("Testing...\n");
    for (int i = 0; i < testTimes; i++)
    {
        //srand()函数产生一个以当前时间开始的随机种子
        srand((unsigned)time(NULL));
        vector<vector<int>> lines = generateLines(maxN,minL,maxR);
        int ans1 = worseSolution(lines);
        int ans2 = heapSolution(lines);
        if (ans1 != ans2)
        {
            printf("%d\n", ans1);
            printf("%d\n", ans2);
            succeed = false;
            break;
        }
    }
    printf(succeed ? "Nice!\n" : "Fucking fucked!\n");    
    return 0;
}