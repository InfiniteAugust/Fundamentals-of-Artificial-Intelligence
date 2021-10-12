load('label.mat')
cd data
%%read 4000 pictures into workspace as training data
train_feature = cell(1,4000);

for one = 1:9 
mone = strcat('0000', num2str(one), '.png');
train_feature{one} = imread(mone);
end
for ten = 10:99
mten = strcat('000', num2str(ten), '.png');
train_feature{ten} = imread(mten);
end
for hundred = 100:999
mhundred = strcat('00', num2str(hundred), '.png');
train_feature{hundred} = imread(mhundred);
end
for thousand = 1000:4000
mthousand = strcat('0', num2str(thousand), '.png');
train_feature{thousand} = imread(mthousand);
end

for k = 1:4000
train_feature{k} = reshape(train_feature{k}, [784, 1]);
end

 train_feature = cell2mat(train_feature);
 train_feature = double(train_feature);
 
%%read 4000 training labels corresponding to the trainin data

train_label = cell(1,4000);

for num = 1:4000
	target = label(num);
		for i = 1:target
			train_label{num}(i, 1) = 0;
		end
		train_label{num}(target + 1, 1) = 1;
        for j = target + 2:10
			train_label{num}(j, 1) = 0;
        end
end

train_label = cell2mat(train_label);

%%read 1000 pictures for testing 
test_feature = cell(1,1000);

for thousand = 4001:5000
mthousand = strcat('0', num2str(thousand), '.png');
test_feature{thousand - 4000} = imread(mthousand);
end

for k = 1:1000
test_feature{k} = reshape(test_feature{k}, [784, 1]);
end

 test_feature = cell2mat(test_feature);
 test_feature = double(test_feature);
 
%%read 1000 label corresponding to the testing data 
 test_label = cell(1,1000);

for num = 1:1000
	target = label(num + 4000);
		for i = 1:target
			test_label{num}(i, 1) = 0;
		end
		test_label{num}(target + 1, 1) = 1;
        for j = target + 2:10
			test_label{num}(j, 1) = 0;
        end
end

test_label = cell2mat(test_label);
%%end of data reading 

%%training and testing 
%%set up the network, adjust parameters and train
myNet = newff(train_feature, train_label, [15, 8]);

myNet.trainParam.goal = 0.01;
myNet.trainParam.epochs = 1500;
myNet.trainParam.lr = 0.05;
myNet.trainParam.max_fail = 5;

myNet = train(myNet, train_feature, train_label);

result = myNet(test_feature);
plotconfusion(test_label, result);

save ann_model.mat;