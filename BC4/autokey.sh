ssh-keygen -t rsa -b 2048
echo "Thank you for using autokey service :)"
read -p "Please enter your username (eg. mw18386): " username 
if [ "$username" == "" ]; then
    echo "Cannot read username"
    exit 1
elif [ ${#username} != 7 ];then
    echo "Please enter the correct username"
    exit 1
fi
echo "Copy your keys to the target server and please enter your usual password to login in"
ssh-copy-id $username@bc4login.acrc.bris.ac.uk
echo "Try to login"
read -p "Do you want to login in now? (y or n)" yesornot
if [ "$yesornot" != "y" ]; then
    echo "You can try to use: ssh $username@bc4login.acrc.bris.ac.uk to login in next time"
    exit 1
fi
ssh $username@bc4login.acrc.bris.ac.uk

