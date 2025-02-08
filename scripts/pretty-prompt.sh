# Prompt

#!/bin/bash

# Useful links:
#   - Credit https://iqcode.com/code/shell/bash-colors
#   - Enhanced with https://github.com/franko/bash-git-prompt/blob/main/git-prompt-linux.sh
#   - https://unix.stackexchange.com/questions/28827/why-is-my-bash-prompt-getting-bugged-when-i-browse-the-history

OFF="\[\e[0m\]"       # Text Reset

# Regular Colors
BLACK='\[\e[0;30m\]'        # Black
RED='\[\e[0;31m\]'          # Red
GREEN='\[\e[0;32m\]'        # Green
YELLOW='\[\e[0;33m\]'       # Yellow
BLUE='\[\e[0;34m\]'         # Blue
PURPLE='\[\e[0;35m\]'       # Purple
CYAN='\[\e[0;36m\]'         # Cyan
WHITE='\[\e[0;37m\]'        # White

# Bold
BBLACK='\[\e[1;30m\]'       # Black
BRED='\[\e[1;31m\]'         # Red
BGREEN='\[\e[1;32m\]'       # Green
BYELLOW='\[\e[1;33m\]'      # Yellow
BBLUE='\[\e[1;34m\]'        # Blue
BPURPLE='\[\e[1;35m\]'      # Purple
BCYAN='\[\e[1;36m\]'        # Cyan
BWHITE='\[\e[1;37m\]'       # White

# Underline
UBLACK='\[\e[4;30,m\]'       # Black
URED='\[\e[4;31,m\]'         # Red
UGREEN='\[\e[4;32,m\]'       # Green
UYELLOW='\[\e[4;33,m\]'      # Yellow
UBLUE='\[\e[4;34,m\]'        # Blue
UPURPLE='\[\e[4;35,m\]'      # Purple
UCYAN='\[\e[4;36,m\]'        # Cyan
UWHITE='\[\e[4;37,m\]'       # White


# Background
ON_DARK='\[\e[48;2;28;28;46m\]'
ON_WAVEBLUE1='\[\e[48;2;34;50;73m\]'

ON_BLACK='\[\e[40,m\]'       # Black
ON_RED='\[\e[41,m\]'         # Red
ON_GREEN='\[\e[42,m\]'       # Green
ON_YELLOW='\[\e[43,m\]'      # Yellow
ON_BLUE='\[\e[44,m\]'        # Blue
ON_PURPLE='\[\e[45,m\]'      # Purple
ON_CYAN='\[\e[46,m\]'        # Cyan
ON_WHITE='\[\e[47,m\]'       # White

# High Intensity
IBLACK='\[\e[0;90,m\]'       # Black
IRED='\[\e[0;91,m\]'         # Red
IGREEN='\[\e[0;92,m\]'       # Green
IYELLOW='\[\e[0;93,m\]'      # Yellow
IBLUE='\[\e[0;94,m\]'        # Blue
IPURPLE='\[\e[0;95,m\]'      # Purple
ICYAN='\[\e[0;96,m\]'        # Cyan
IWHITE='\[\e[0;97,m\]'       # White

# Bold High Intensity
BIBLACK='\[\e[1;90m\]'      # Black
BIRED='\[\e[1;91m\]'        # Red
BIGREEN='\[\e[1;92m\]'      # Green
BIYELLOW='\[\e[1;93m\]'     # Yellow
BIBLUE='\[\e[1;94m\]'       # Blue
BIPURPLE='\[\e[1;95m\]'     # Purple
BICYAN='\[\e[1;96m\]'       # Cyan
BIWHITE='\[\e[1;97m\]'      # White

# High Intensity backgrounds
ON_IBLACK='\[\e[0;100,m\]'   # Black
ON_IRED='\[\e[0;101,m\]'     # Red
ON_IGREEN='\[\e[0;102,m\]'   # Green
ON_IYELLOW='\[\e[0;103,m\]'  # Yellow
ON_IBLUE='\[\e[0;104,m\]'    # Blue
ON_IPURPLE='\[\e[0;105,m\]'  # Purple
ON_ICYAN='\[\e[0;106,m\]'    # Cyan
ON_IWHITE='\[\e[0;107,m\]'   # White



if [[ "$OSTYPE" == "darwin"* ]]; then
  UNICODE_BRANCH_SYMBOL="λ"
else
  UNICODE_BRANCH_SYMBOL=$(echo -e "\ue0a0")
fi

git_branch_name () {
  local xroot="$PWD"
  while [ ! -e "${xroot}/.git" ]; do
    local updir=${xroot%/*}
    if [ "$updir" == "$xroot" ]; then
      break
    fi
    xroot="$updir"
  done
  local xgit_dir="$xroot/.git"
  local worktree_tag
  local fline
  if [ -f "$xgit_dir" ]; then
    # if .git is a file it can be a worktree
    read fline < "$xgit_dir"
    xgit_dir="${fline#gitdir: }"
    worktree_tag=" (worktree)"
  fi
  if [ -f "$xgit_dir/HEAD" ]; then
    read fline < "$xgit_dir/HEAD"
    printf "${fline##*/}$worktree_tag"
  fi
}

#PS1_BACKGR=$ON_WAVEBLUE1
PS1_BACKGR=""
PS1_PROMPT_NL="" # space for 1-line prompt, \n for double-line prompt
PS1_LOW=${WHITE}
PS1_USER=$PURPLE$PS1_BACKGR
PS1_HOST=$BPURPLE$PS1_BACKGR
PS1_PLAIN=$BWHITE$PS1_BACKGR
PS1_PATH=$WHITE$PS1_BACKGR
PS1_GIT_DIRTY=$BYELLOW$PS1_BACKGR
PS1_GIT_CLEAN=$BGREEN$PS1_BACKGR

if [ -f ~/.config/ps1-vars ]; then
    . ~/.config/ps1-vars
fi

export PROMPT_DIRTRIM=3

update_ps1_git() {
  if [[ -n $(git status -s 2> /dev/null) ]]; then
    PS1_GIT_BRANCH="${PS1_GIT_DIRTY}"
  else
    PS1_GIT_BRANCH=${PS1_GIT_CLEAN}
  fi

  BRANCH="$(git_branch_name)"
  if [ ! -z "${BRANCH}" ]; then
    BRANCH="${PS1_PLAIN}${UNICODE_BRANCH_SYMBOL} ${PS1_GIT_BRANCH}${BRANCH} "
  fi
  #export PS1="${PS1_LOW}╭┤ ${PS1_USER}\u${PS1_PLAIN}@${PS1_HOST}\h ${BRANCH}${PS1_PATH}\w${PS1_PLAIN}${PS1_PROMPT_NL}${PS1_LOW}╰─${OFF}$ "
  #export PS1="${PS1_LOW}${PS1_USER}\u${PS1_PLAIN}@${PS1_HOST}\h ${BRANCH}${PS1_PATH}\w${PS1_PLAIN}${PS1_LOW}${OFF}$ "
  export PS1="${PS1_LOW}${PS1_USER}\u${PS1_PLAIN}@${PS1_HOST}\h ${BRANCH}${PS1_PATH}\w${PS1_PLAIN}${PS1_PROMPT_NL}${PS1_LOW}${OFF} $ "
}

update_ps1() {
  export PS1="${PS1_BACKGR}${PS1_USER}\u${PS1_PLAIN}@${PS1_HOST}\h ${PS1_PATH}\w${PS1_PLAIN}${PS1_PROMPT_NL}${OFF}$ "
}

PROMPT_DIRTRIM=2
PROMPT_COMMAND="update_ps1_git; history -a; history -c; history -r"