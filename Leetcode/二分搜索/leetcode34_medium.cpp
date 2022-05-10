class Solution {
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        int left = 0, right = nums.size()-1; // vector长度用size
        // 左右位置
        vector<int> index(2);
        // 左边界
        while(left<=right){
            // 在循环内求中点
            int mid =  left + (right-left)/2;
            if(nums[mid]==target){
                right = mid - 1;
            }else if(nums[mid]>target){
                right = mid - 1;
            }else if(nums[mid]<target){
                left = mid + 1;
            }
        }
        if(left>=nums.size() || nums[left]!=target){
            index[0]=-1;
        }else{
            index[0]=left;
        }
        
        // 右边界
        left = 0, right = nums.size()-1; // vector长度用size
        while(left<=right){
            // 在循环内求中点
            int mid =  left + (right-left)/2;
            if(nums[mid]==target){
                left = mid + 1;
            }else if(nums[mid]>target){
                right = mid - 1;
            }else if(nums[mid]<target){
                left = mid + 1;
            }
        }

        if(right<0 || nums[right]!=target){
            index[1]=-1;
        }else{
            index[1]=right;
        }   
        return index;
    }
};