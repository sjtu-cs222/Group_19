from VGG16_mnist import *
import pickle as pkl
def test(number,model,test_data_loader):
    # set model to evaluating/testing mode
    model.eval()
    correct = 0
    total = 0
    Turetotal = 0
    Judgetotal = 0
    recog = 0
    
    for i, (batch, label) in enumerate(test_data_loader):
#         if self.use_gpu:
        batch = batch.cuda()
#         if self.torch_version == 0.3:
#             batch = Variable(batch)
        output = model(batch)
#         print(output.data)
# #         break
        pred = output.data.max(1)[1]
        label = label==number
        pred = pred==number
        correct += pred.cpu().eq(label).sum()
        total += label.size(0)
        Turetotal += pred.cpu().sum()
        Judgetotal+= label.sum()
        #print(pred)
        #print(label)
        recog += (pred.cpu() & label.cpu()).sum()

    print("Accuracy : %f" % (float(correct) / total))
    print("correct:", correct, "total:", total,"Turetotal:",Turetotal, "Judgetotal:",Judgetotal,"regoc",recog )
    print("pricision:",float(recog)/Turetotal,"recall",float(recog)/Judgetotal)
    # set model return to training mode
    # self.model.train()
    return (float(correct)/total, correct, total, Turetotal, Judgetotal, recog )

if __name__ == "__main__":
    
    savedStdout = sys.stdout

    with open('./vggtest_out_mnist.txt', 'w+', 1) as redirect_out:
        dataset=DataSet(torch_v=0.4)
        test_data_loader = dataset.test_loader("../mnist_data",pin_memory=True)
        res = {}
        for n in range(10):
            print("model:",n)
            res[n] = []
            for i in range(8):
                model = torch.load("./pruned/vgg16_mnist_"+str(n)+"_prunned_"+str(i)).cuda()
                res[n].append(test(n,model,test_data_loader))
    with open("testresult.pkl","wb") as f:
        pkl.dump(res,f)
    
    sys.stdout = savedStdout
