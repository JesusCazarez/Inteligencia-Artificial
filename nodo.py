class Nodo:
    def __init__(self, valor):
        self.valor = valor
        self.izquierdo = None
        self.derecho = None

class ArbolBinario:
    def __init__(self):
        self.raiz = None

    def insertar(self, valor):
        self.raiz = self._insertar_rec(self.raiz, valor)

    def _insertar_rec(self, nodo, valor):
        if nodo is None:
            return Nodo(valor)

        if valor < nodo.valor:
            nodo.izquierdo = self._insertar_rec(nodo.izquierdo, valor)
        elif valor > nodo.valor:
            nodo.derecho = self._insertar_rec(nodo.derecho, valor)

        return nodo

    def imprimir_arbol(self):
        self._imprimir_arbol_rec(self.raiz, 0)

    def _imprimir_arbol_rec(self, nodo, nivel):
        if nodo is not None:
            self._imprimir_arbol_rec(nodo.derecho, nivel + 1)
            print(" " * (nivel * 4) + str(nodo.valor))
            self._imprimir_arbol_rec(nodo.izquierdo, nivel + 1)

# Prueba del Árbol en Python
if __name__ == "__main__":
    arbol = ArbolBinario()
    arbol.insertar(25)
    arbol.insertar(37)
    arbol.insertar(87)
    arbol.insertar(10)
    arbol.insertar(6)
    arbol.insertar(98)
    arbol.insertar(50)
    
    print("Árbol Binario de Búsqueda:")
    arbol.imprimir_arbol()