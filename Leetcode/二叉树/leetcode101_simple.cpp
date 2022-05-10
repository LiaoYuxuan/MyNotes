// 循环法
class Solution {
public:
    bool isSymmetric(TreeNode* root) {
        // 思路一：写出二叉树的中序遍历，然后看是否对称(不全正确：因为有可能不对称，但中序遍历对称)
        // 思路二：将原本的树整体做镜像对称得到另一颗树，用队列去记录对比层序遍历结果

        //只有一个节点，则对称
        if(root->left == nullptr && root->right == nullptr) return true;
        //得到中序遍历结果，并判断奇偶性
        queue<TreeNode*> q;
        TreeNode* lp = root;
        TreeNode* rp = root;
        //最初把根节点入队两次，作为两棵树
        q.push(lp);
        q.push(rp);

        while(q.size()!=0){
            //同时出队两个节点，判断值是否一样
            lp=q.front();
            q.pop();
            rp=q.front();
            q.pop();
            if(lp->val != rp->val) return false;
            //左树的左子节点和右树的右子节点入队
            if(lp->left!=nullptr && rp->right!=nullptr){
                q.push(lp->left);
                q.push(rp->right);
            }else if(lp->left==nullptr && rp->right==nullptr){
            }else return false;
            //左树的右子节点和右树的左子节点入队
            if(lp->right!=nullptr && rp->left!=nullptr){
                q.push(lp->right);
                q.push(rp->left);
            }else if(lp->right==nullptr && rp->left==nullptr){
            }else return false;
        }
        return true;
    }
};

// 递归法
class Solution {
public:
    bool isSymmetric(TreeNode* root) {
        return traverse(root->left,root->right);
    }

    bool traverse(TreeNode* a, TreeNode* b) {
        if(a==nullptr && b==nullptr) return true;
        else if(a!=nullptr && b!=nullptr){
            if(a->val!=b->val) return false;
            else{
                return traverse(a->left,b->right) && traverse(a->right,b->left);
            }
        }else return false;
    }
};