% Cargar series originales sin la primera fila (encabezados)
datos_originales = csvread('originales.csv', 1, 0);

% Elegir el índice de la serie que deseas visualizar
indice_serie = 18;

% Identificar escalones y valores anómalos en la serie original
diferencias = diff(datos_originales(:, indice_serie));
escalones = find(abs(diferencias) > 200); % Identificar escalones con magnitud mayor a 200
valores_anomalos = find(abs(diferencias) > 100 & abs(diferencias) <= 200); % Identificar valores anómalos con magnitud entre 100 y 200

% Visualizar la serie original
figure;
plot(datos_originales(:, indice_serie), 'b', 'LineWidth', 2);
hold on;

% Resaltar los escalones con cruces azules oscuro
plot(escalones, datos_originales(escalones, indice_serie), 'x', 'MarkerSize', 10, 'Color', [0 0 0.5]);
legend('Original', 'Escalones');

% Resaltar los valores anómalos con círculos azules oscuro
plot(valores_anomalos, datos_originales(valores_anomalos, indice_serie), 'o', 'MarkerSize', 10, 'Color', [0 0 0.5]);
legend('Original');

hold off;

xlabel('Índice');
ylabel('Valor');
title(sprintf('Serie %d - Original', indice_serie));

