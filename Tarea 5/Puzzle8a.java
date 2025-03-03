import javax.swing.*;
import java.awt.*;
import java.util.*;
import java.util.List;

public class Puzzle8a extends JFrame {
    private JButton[][] botones = new JButton[3][3];
    private int[][] tablero = new int[3][3];
    private final int VACIO = 0;
    private int contadorMovimientos = 0;
    private JLabel etiquetaMovimientos;
    private JPanel panelJuego;
    private JButton botonResolver;
    private JButton botonReiniciar;
    private JButton botonConfigurar;

    public Puzzle8a() {
        setTitle("Juego de Puzzle 8");
        setSize(500, 600);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLayout(new BorderLayout());

        etiquetaMovimientos = new JLabel("Movimientos: 0", SwingConstants.CENTER);
        etiquetaMovimientos.setFont(new Font("Arial", Font.BOLD, 20));
        add(etiquetaMovimientos, BorderLayout.NORTH);

        panelJuego = new JPanel(new GridLayout(3, 3, 5, 5));
        panelJuego.setBorder(BorderFactory.createEmptyBorder(20, 20, 20, 20));
        inicializarTablero();
        add(panelJuego, BorderLayout.CENTER);

        JPanel panelBotones = new JPanel(new GridLayout(1, 3, 10, 10));
        panelBotones.setBorder(BorderFactory.createEmptyBorder(10, 20, 20, 20));

        botonConfigurar = new JButton("Configurar Tablero");
        botonConfigurar.addActionListener(e -> configurarTablero());
        panelBotones.add(botonConfigurar);

        botonResolver = new JButton("Resolver");
        botonResolver.addActionListener(e -> {
            if (esObjetivo(tablero)) {
                JOptionPane.showMessageDialog(this, "El puzzle ya está resuelto.");
            } else if (!esResoluble(tablero)) {
                JOptionPane.showMessageDialog(this, "El tablero no tiene solución.");
            } else {
                new Thread(this::resolverPuzzle).start();
            }
        });
        panelBotones.add(botonResolver);

        botonReiniciar = new JButton("Reiniciar");
        botonReiniciar.addActionListener(e -> reiniciarTablero());
        panelBotones.add(botonReiniciar);

        add(panelBotones, BorderLayout.SOUTH);

        setVisible(true);
    }

    private void configurarTablero() {
        JPanel panel = new JPanel(new GridLayout(3, 3, 5, 5));
        JTextField[][] campos = new JTextField[3][3];

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                campos[i][j] = new JTextField(2);
                campos[i][j].setFont(new Font("Arial", Font.PLAIN, 20));
                campos[i][j].setHorizontalAlignment(JTextField.CENTER);
                campos[i][j].setText(tablero[i][j] == VACIO ? "" : String.valueOf(tablero[i][j]));
                panel.add(campos[i][j]);
            }
        }

        int resultado = JOptionPane.showConfirmDialog(
                this,
                panel,
                "Configurar Tablero",
                JOptionPane.OK_CANCEL_OPTION,
                JOptionPane.PLAIN_MESSAGE
        );

        if (resultado == JOptionPane.OK_OPTION) {
            int[][] nuevoTablero = new int[3][3];
            try {
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        String texto = campos[i][j].getText().trim();
                        nuevoTablero[i][j] = texto.isEmpty() ? VACIO : Integer.parseInt(texto);
                    }
                }
                if (validarTablero(nuevoTablero)) {
                    tablero = nuevoTablero;
                    actualizarBotones();
                } else {
                    JOptionPane.showMessageDialog(this, "El tablero ingresado no es válido.");
                }
            } catch (NumberFormatException e) {
                JOptionPane.showMessageDialog(this, "Ingrese solo números válidos.");
            }
        }
    }

    private boolean validarTablero(int[][] tablero) {
        boolean[] numeros = new boolean[9];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                int valor = tablero[i][j];
                if (valor < 0 || valor > 8 || numeros[valor]) {
                    return false;
                }
                numeros[valor] = true;
            }
        }
        return true;
    }

    private void inicializarTablero() {
        int[] numeros = {1, 2, 3, 4, 5, 6, 7, 8, VACIO};
        do {
            mezclarArray(numeros);
        } while (!esResoluble(numeros)); // Asegura que el tablero sea resoluble

        int indice = 0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                tablero[i][j] = numeros[indice++];
                botones[i][j] = new JButton(tablero[i][j] == VACIO ? "" : String.valueOf(tablero[i][j]));
                botones[i][j].setFont(new Font("Arial", Font.BOLD, 30));
                botones[i][j].setFocusable(false);
                panelJuego.add(botones[i][j]);
            }
        }
        contadorMovimientos = 0;
        actualizarBotones();
    }

    private void reiniciarTablero() {
        panelJuego.removeAll();
        inicializarTablero();
        panelJuego.revalidate();
        panelJuego.repaint();
    }

    private boolean esResoluble(int[] numeros) {
        int inversiones = 0;
        for (int i = 0; i < numeros.length; i++) {
            for (int j = i + 1; j < numeros.length; j++) {
                if (numeros[i] != VACIO && numeros[j] != VACIO && numeros[i] > numeros[j]) {
                    inversiones++;
                }
            }
        }
        return inversiones % 2 == 0; // El puzzle es resoluble si el número de inversiones es par
    }

    private boolean esResoluble(int[][] tablero) {
        int[] numeros = new int[9];
        int indice = 0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                numeros[indice++] = tablero[i][j];
            }
        }
        return esResoluble(numeros);
    }

    private void mezclarArray(int[] array) {
        Random rand = new Random();
        for (int i = array.length - 1; i > 0; i--) {
            int j = rand.nextInt(i + 1);
            int temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
    }

    private void actualizarBotones() {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                botones[i][j].setText(tablero[i][j] == VACIO ? "" : String.valueOf(tablero[i][j]));
            }
        }
        etiquetaMovimientos.setText("Movimientos: " + contadorMovimientos);
    }

    private void resolverPuzzle() {
        List<int[][]> solucion = calcularSolucion();
        if (solucion != null) {
            for (int[][] estado : solucion) {
                tablero = estado;
                contadorMovimientos++;
                actualizarBotones();
                try {
                    Thread.sleep(500);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
            JOptionPane.showMessageDialog(this, "¡Puzzle resuelto en " + contadorMovimientos + " movimientos!");
        } else {
            JOptionPane.showMessageDialog(this, "No se encontró solución.");
        }
    }

    private List<int[][]> calcularSolucion() {
        PriorityQueue<Nodo> cola = new PriorityQueue<>(Comparator.comparingInt(n -> n.costo));
        Set<String> visitados = new HashSet<>();
        cola.add(new Nodo(tablero, null, 0, heuristica(tablero)));

        while (!cola.isEmpty()) {
            Nodo actual = cola.poll();
            if (esObjetivo(actual.estado)) {
                return reconstruirCamino(actual);
            }
            visitados.add(Arrays.deepToString(actual.estado));
            for (int[][] vecino : obtenerMovimientos(actual.estado)) {
                if (!visitados.contains(Arrays.deepToString(vecino))) {
                    cola.add(new Nodo(vecino, actual, actual.costo + 1, heuristica(vecino)));
                }
            }
        }
        return null;
    }

    private int heuristica(int[][] estado) {
        int distancia = 0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                int valor = estado[i][j];
                if (valor != VACIO) {
                    int objetivoX = (valor - 1) / 3;
                    int objetivoY = (valor - 1) % 3;
                    distancia += Math.abs(i - objetivoX) + Math.abs(j - objetivoY);
                }
            }
        }
        return distancia;
    }

    private boolean esObjetivo(int[][] estado) {
        int contador = 1;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (i == 2 && j == 2) {
                    if (estado[i][j] != VACIO) return false;
                } else if (estado[i][j] != contador) {
                    return false;
                }
                contador++;
            }
        }
        return true;
    }

    private List<int[][]> obtenerMovimientos(int[][] estado) {
        List<int[][]> vecinos = new ArrayList<>();
        int[] dx = { -1, 1, 0, 0 }; // Movimientos en filas (arriba, abajo)
        int[] dy = { 0, 0, -1, 1 }; // Movimientos en columnas (izquierda, derecha)

        // Encuentra la posición del espacio vacío (0)
        int x = -1, y = -1;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (estado[i][j] == VACIO) {
                    x = i;
                    y = j;
                    break;
                }
            }
            if (x != -1) break;
        }

        // Genera los movimientos posibles
        for (int i = 0; i < 4; i++) {
            int nx = x + dx[i];
            int ny = y + dy[i];

            if (nx >= 0 && nx < 3 && ny >= 0 && ny < 3) {
                int[][] nuevoEstado = new int[3][3];
                for (int j = 0; j < 3; j++) {
                    System.arraycopy(estado[j], 0, nuevoEstado[j], 0, 3);
                }
                // Intercambia el espacio vacío con el vecino
                nuevoEstado[x][y] = nuevoEstado[nx][ny];
                nuevoEstado[nx][ny] = VACIO;
                vecinos.add(nuevoEstado);
            }
        }

        return vecinos;
    }

    private List<int[][]> reconstruirCamino(Nodo nodo) {
        List<int[][]> camino = new ArrayList<>();
        while (nodo != null) {
            camino.add(0, nodo.estado);
            nodo = nodo.padre;
        }
        return camino;
    }

    private static class Nodo {
        int[][] estado;
        Nodo padre;
        int costo;
        int heuristica;

        Nodo(int[][] estado, Nodo padre, int costo, int heuristica) {
            this.estado = estado;
            this.padre = padre;
            this.costo = costo + heuristica;
        }
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(Puzzle8a::new);
    }
}