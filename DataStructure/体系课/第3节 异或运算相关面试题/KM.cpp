#include<iostream>
#include<vector>
#include <unordered_map>
#include <unordered_set>
#include <time.h>
using namespace std;

// 利用异或运算求解KM问题
// 建立移位运算（如0100）与t数组下标（0100是第2位为1，所以0100要映射为2）之间映射的哈希表
// 这里一定要加&，因为要改变Hashmap
void creatmap(unordered_map<int,int> &Hashmap){
    for (int i = 0; i < 32; i++){
       Hashmap[1 << i] = i;
       // 另一种插入方法
       // Hashmap.insert(make_pair(1 << i,i));
    }
}

int KM(vector<int> arr, int K, int M){
    unordered_map<int,int> Hashmap;
    if (Hashmap.empty()){
        creatmap(Hashmap);
    }

    // 第一步：设置一个长度为32的数组，并初始化为0
    vector<int> t(32, 0);
    // 第二步：遍历数组中的每个数
    for (auto it = arr.begin(); it != arr.end(); it++){
        int num = *it;
        while (num != 0){
            // 依次找出最右边的1，在t中对应位置处加1
            int right = num & (-num);
            // 注意find方法返回的是一个迭代器，需要正确使用
            t[Hashmap.find(right)->second] += 1;
            // 消去最右边的1
            num ^= right;
        }
    }
    //第三步：判断t数组中各个元素的组成
    int ans = 0;
    for (int i = 0; i < 32; i++){
        if(t[i] % M == 0){
            continue;
        }
        if (t[i] % M == K)     ans |= (1<<i);
        else    return -1;
    }
    if (ans == 0){
        int flag = 0;
        for (auto it = arr.begin(); it != arr.end(); it++){
            if (*it==0) flag++;
        }
        if (flag != K)    return -1;        
    }
    return ans;
}

// 对数器
int comparator(vector<int> arr, int K, int M){
    unordered_map<int,int> Hashmap;
    for (auto it = arr.begin(); it != arr.end(); it++){
        // 已存在
        if (Hashmap.find(*it) != Hashmap.end())    Hashmap[*it] +=1;
        else    Hashmap[*it] =1;
    }
    for (auto it = Hashmap.begin(); it != Hashmap.end(); it++) {
        if (it->second == K)    return it->first;  
    }
    return -1;
}

// 交换函数
void swap(vector<int> &arr, int index1, int index2){
    int tmp = arr[index1];
    arr[index1] = arr[index2];
    arr[index2] = tmp;
}

// 生成符合要求的随机数组
vector<int> generateRandomArray(int maxKinds, int maxValue, int K, int M){
        // 首先生成数据的种数（最少2种，最多maxKinds种）
        int kinds = (rand()%(maxKinds-2+1))+2;
        // 有可能生成的数组不满足K的要求
        int K_times = rand()/double(RAND_MAX) < 0.5? K:(rand()%(M-1))+1;
        vector<int> arr;
        // vector<int> arr(1*K_times+(kinds-1)*M);
        // 然后生成对应kinds个不同的数据，利用哈希集合判断是否重复
        unordered_set<int> Hashset;
        for (int i = 0; i < kinds; i++){
            int num = 0;
            do{
                num = (rand()%(2*maxValue+1))-maxValue;  
                // 为了避免下一次产生的数据还重复，用while结构
            } while (Hashset.find(num) != Hashset.end());
            Hashset.emplace(num);
        }
        // 先将K个相同的数放进去, 这里有问题，先前生成了指定大小vector后就不能用push_back了
        for (int i = 0; i < K_times; i++){
            arr.push_back(*(Hashset.begin()));
        }
        // 再将M个相同的其他数放进去
        for (auto it = ++Hashset.begin(); it != Hashset.end(); it++){
            for (int i = 0; i < M; i++){
                arr.push_back(*it);
            }
        }
        //随机交换i和j处的值，打乱数组
        for (int i = 0; i < arr.size(); i++){
            int j = rand()%(arr.size());
            swap(arr, i, j);
        }
        return arr;
}

// 顺序输出数组的函数
void printArray(vector<int> arr){
    // 使用迭代器的方式访问vector，这里arr.begin()和arr.end()都是指针, auto是vector<int>::iterator的简写形式
    for (auto it = arr.begin(); it != arr.end(); it++)
    {
        printf("%d ",*it);
    }
    cout << endl;
}

// 测试使用
void test(){
        // 最多有多少种数字
        int maxKinds = 10;
        // 每个数字的范围
        int maxValue = 30;
		int testTime = 500000;
        // K和M的最大范围
		int maxKM = 9;
        bool succeed = true;
        //srand()函数产生一个以当前时间开始的随机种子
        srand((unsigned)time(NULL));
		for (int i = 0; i < testTime; i++) {
            // (rand()%(b-a+1))+a: 获得[a,b]的随机整数，这里取a=1,b=maxKM
            int a = (rand()%maxKM)+1;
            int b = (rand()%maxKM)+1;
            int K = min(a, b);
            int M = max(a, b);
            // 如果产生的随机数相同，则+1
            if(K == M)    M += 1;
            vector<int> arr = generateRandomArray(maxKinds, maxValue, K, M);
			if (KM(arr, K, M) != comparator(arr, K, M)) {
				succeed = false;
				printArray(arr);
                printf("%d\n", KM(arr, K, M));
                printf("%d\n", comparator(arr, K, M));
				break;
			}
		}
        printf(succeed ? "Nice!\n" : "Fucking fucked!\n");    
}

int main(){
    test();
}