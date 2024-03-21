var cnv;
let x_vals = [0.5];
let y_vals = [0.5];
let m, b;
// learning rate
const lr = 0.1;
// optimizer
const opt = tf.train.sgd(lr);

function setup() {
  cnv = createCanvas(windowWidth-100, windowHeight-100);
  cnv.position((windowWidth-width)/2, (windowHeight-height)/2);
  m = tf.variable(tf.scalar(random(1)));
  b = tf.variable(tf.scalar(random(1)));
  
}

function loss(pred, labels) {
  // take predictions, sub difference from true values, square them, and take the mean
  const loss = pred.sub(labels).square().mean();
  // loss.print();
  return loss;
}

// using x vals, make a tensor of x vals, and return their respective y vals in a tensor
function predict(x) {
  const tx = tf.tensor1d(x);
  const ty = tx.mul(m).add(b);
  return ty;
}

function draw() {
  let loss_val;
  if (x_vals.length > 0) {
  // optimize
    const ty = tf.tensor1d(y_vals);
    opt.minimize(() => loss(predict(x_vals), ty));
    loss_val = loss(predict(x_vals), ty);
  }
  background(0);
  stroke(255);
  strokeWeight(6);
  for (let i = 0; i < x_vals.length; i++) {
    point(map(x_vals[i], 0, 1, 0, width), map(y_vals[i], 0, 1, height, 0));
  }
  strokeWeight(4);
  stroke(color(70, 230, 70));
  const xd = [0, 1];
  const yd = predict(xd).dataSync();
  let x1 = xd[0] * width;
  let y1 = map(yd[0], 0, 1, height, 0);
  let x2 = xd[1] * width;
  let y2 = map(yd[1], 0, 1, height, 0);
  line(x1, y1, x2, y2);
  fill(0);
  noStroke();
  fill(255);
  text("Loss", 30, 25)
  text(loss_val.dataSync()[0].toFixed(6), 30, 40);
  stroke(255);
  noFill();
  rect(0, 0, width, height);
}

function mousePressed() {
  x_vals.push(map(mouseX, 0, width, 0, 1));
  y_vals.push(map(mouseY, 0, height, 1, 0));
}
