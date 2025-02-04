
1.
Lo primero que hacemos en el codigo es declarar la clase nodo, 
en esta simplemente declaramos el constructor del nodo, 
dandole el valor al nodo, y declarando vacio el izquierdo 
y el derecho por que aun no tienen hermanos.

2.
Tenemos la clase ArbolBinario, en el constructor declaramos
la raiz como vacia por que asi comienza.

3.
En el metodo insertar, creamos la raiz en caso de que el 
arbol esté vacio, y si ya tiene elementos, llama al siguiente
metodo, insertar recursivo para enocontrar la posicion.

4.
El metodo insertar recursivo, si el nodo se encuentra vacio,
creamos un nodo nuevo, y revisa si el valor es mayor o menor,
si es menor baja por la izquierda, en caso contrario por
la derecha, sigue asi hasta que encuentra el lugar vacio 
correcto.

5.
Los metodos de imprimir lo que hacen es imprimir los valores
de los numeros empezando por la derecha con los numeros 
grandes.