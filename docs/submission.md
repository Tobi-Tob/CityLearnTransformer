# Making submission

This file will help you in making your first submission.


## Submission Entrypoint

The evaluator will create an instance of `OrderEnforcingAgent` from `orderenforcingwrapper.py` to run the evaluation. 

## Where to write my code?

Please refer to [How to write your own agent?](../readme.md#how-to-write-your-own-agent) and [How to write your own reward function?](../readme.md#how-to-write-your-own-reward-function) sections in the [readme.md](../readme.md)
   

Once done, you can specify the agent you want to run in `agents/user_agent.py` and reward function in `rewards/get_reward.py`.

## Setting up SSH keys

You will have to add your SSH Keys to your GitLab account by going to your profile settings [here](https://gitlab.aicrowd.com/profile/keys). If you do not have SSH Keys, you will first need to [generate one](https://docs.gitlab.com/ee/ssh/README.html#generating-a-new-ssh-key-pair).


## Submit using the utility script

You can run the following to make a submission.

```bash
./submit.sh <description-phrase> # example: ./submit.sh submission-my-first-agent 
```

`./submit.sh` contains a few git commands that will push your code to AIcrowd GitLab.

**Note:** In case you see an error message like `git: 'lfs' is not a git command. See 'git --help'.`, please make sure you have git LFS installed. You can install it using `git lfs install` or refer [Git LFS website](https://git-lfs.github.com/).

## Pushing the code manually

### IMPORTANT: Saving Models before submission!

Before you submit make sure that you have saved your models, which are needed by your inference code.
In case your files are larger in size you can use `git-lfs` to upload them. More details [here](https://discourse.aicrowd.com/t/how-to-upload-large-files-size-to-your-submission/2304).

## How to submit your code?

You can create a submission by making a _tag push_ to your repository on [https://gitlab.aicrowd.com/](https://gitlab.aicrowd.com/).
**Any tag push (where the tag name begins with "submission-") to your private repository is considered as a submission**

```bash
cd citylearn-2022-starter-kit

# Add AIcrowd git remote endpoint
git remote add aicrowd git@gitlab.aicrowd.com:<YOUR_AICROWD_USER_NAME>/citylearn-2022-starter-kit.git 
git push aicrowd master
```

```bash

# Commit All your changes
git commit -am "My commit message"

# Create a tag for your submission and push
git tag -am "submission-v0.1" submission-v0.1
git push aicrowd master
git push aicrowd submission-v0.1

# Note : If the contents of your repository (latest commit hash) does not change,
# then pushing a new tag will **not** trigger a new evaluation.
```

You now should be able to see the details of your submission at:
`https://gitlab.aicrowd.com/<YOUR_AICROWD_USER_NAME>/citylearn-2022-starter-kit/issues`

**NOTE**: Please remember to update your username instead of `<YOUR_AICROWD_USER_NAME>` in the above link :wink:

After completing above steps, you should start seeing something like below to take shape in your Repository -> Issues page:
![](https://i.imgur.com/17U52oB.png)

### Other helpful files

👉 [runtime.md](/docs/runtime.md)
