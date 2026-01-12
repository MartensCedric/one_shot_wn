run(strcat('../build/3d/', name, '.m'));
slice = reshape(flip(chis), res);
fig = imagesc(slice);
hold on

caxis([-1, 1]); 
cb = colorbar;
axis equal
axis off
axis tight
exportgraphics(gcf,strcat(name, '.tiff'),'Resolution',800)

