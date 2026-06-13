#include <stdlib.h>

bar connection(){
foo = malloc(1024);
return foo;
}
endConnection(bar foo) {

free(foo);
}

int main() {

while(1) {//thread 1
//On a connection
foo=connection(); //thread 2
//When the connection ends
endConnection(foo)
}
}
