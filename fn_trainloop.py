import torch
from copy import deepcopy

def train_loop(model, trainloader, loss_fn, epochs, optimizer, schduler):
	steps = 0
	steps_per_epoch = len(trainloader)
	min_loss = 100000
	max_accuracy = 0
	trigger = 0
	paitence = 7

	for epoch in range(epochs):
		model.train()
		train_loss = 0

		for batch in trainloader:
			steps += 1
			# 입력 데이터 준비
			images = batch['image']
			labels = batch['label']
			images, labels = images.to(device), labels.to(device)


			# 전방향 예측
			predict = model(images)
			loss = loss_fn(predict, labels)

			# 오차역전파 및 모델파라미터 업데이트
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			train_loss += loss.item()
			if (steps % steps_per_epoch) == 0:
				model.eval()

				valid_loss, valid_accuracy = validate(model, validloader, loss_fn)

				print('-' * 50)
				print('Epoch : {}/{}......'.format(epoch+1, epochs))
				print('Train Loss : {:.4f}'.format(train_loss/len(trainloader)))
				print('Valid Loss : {:.4f}'.format(valid_loss/len(validloader)))
				print('Valid Accuracy : {:.4f}'.format(valid_accuracy))

				if valid_accuracy > max_accuracy:
					max_accuracy = valid_accuracy
					best_model_state = deepcopy(model.state_dict())
					torch.save(best_model_state, 'best_checkpoint.pth')

				if valid_loss > min_loss:
					trigger += 1
					print('trigger :', trigger)

					if trigger > paitence:
						print('Early Stopping !!!')
						print('Training loop is finished')
						return

				else:
					trigger = 0
					min_loss = valid_loss

				schduler.step(valid_loss)
	
	return