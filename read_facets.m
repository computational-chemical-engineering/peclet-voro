cla;
pos=get(gcf,'Position');
pos(3:4)=[384,384];
set(gcf,'Position',pos);
set(gcf,'Color',[1 1 1]);
gcl=camlight;
%axis tight;
axis([0 1 0 1 0 1]);
axis vis3d;
set(gca,'box','on','color',[1 1 1],'Linewidth',2,'projection','perspective','Xcolor',[0 0 0],'Xtick',[],'XtickLabel','','Ycolor',[0 0 0],'Ytick',[],'YtickLabel','','Zcolor',[0 0 0],'Ztick',[],'ZtickLabel','');
 set([findobj(gca,'type','patch');findobj(gca,'type','surface')],...
       'FaceLighting','flat',...
       'AmbientStrength',.9,'DiffuseStrength',.9,...
       'SpecularStrength',.9,'SpecularExponent',2,...
       'BackFaceLighting','unlit');
clear('A', 'F');
A = [];

n = 300;
F(n) = struct('cdata',[],'olormap',[]);

for k=1:n
    s=strcat('00000',num2str(k));
    s=s(length(s)-2:length(s));
    filename = strcat('intf_',s,'.dat');
    fid = fopen(filename);
    cla;
    newline = sprintf('\n');
    line = fgets(fid);
    i = 0;
    j = 1;
    while ischar(line)
        if strcmp(newline, line)
            patch('Xdata',A(1:i,1),'Ydata',A(1:i,2),'Zdata',A(1:i,3),'FaceColor',...
                [0.4 1 0.4],'EdgeColor',[0.2 0.5 0.2],'FaceAlpha',0.7);
            i = 0;
            j = j+1;
        else
            i=i+1;
            A(i,:)=sscanf(line,'%f %f %f');
        end
        line = fgets(fid);
    end
    fclose(fid);
    F(k) = getframe;
end

writerObj = VideoWriter('cube.mp4');
open(writerObj);
writeVideo(writerObj, F(1:n));
close(writerObj);
