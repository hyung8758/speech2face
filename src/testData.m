%% Test the Network
pts = data.pts;
mfcc = data.mfcc;
for idx = 70:80
x = mfcc(:,idx);
y = net(x);

plot(y(1:18),y(19:36),'or');hold on
plot(pts(1:18,idx),pts(19:36,idx),'ob');
axis([-3 3 -3 3]);
hold off
pause(1)
end