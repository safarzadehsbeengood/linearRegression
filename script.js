var cnv;
let x_vals = [0.5];
let y_vals = [0.5];
let m, b, loss_val;
const lr = 0.1;
const opt = tf.train.sgd(lr);

function setup() {
  cnv = createCanvas(windowWidth - 100, windowHeight - 100);
  cnv.position((windowWidth - width) / 2, (windowHeight - height) / 2);
  m = tf.variable(tf.scalar(random(1)));
  b = tf.variable(tf.scalar(random(1)));
}

function loss(pred, labels) {
  // take predictions, sub difference from true values, square them, and take the mean
  const loss = pred.sub(labels).square().mean();
  loss_val = loss.dataSync()[0];
  return loss;
}

// using x vals, make a tensor of x vals, and return their respective y vals in a tensor
function predict(x) {
  const tx = tf.tensor1d(x);
  const ty = tx.mul(m).add(b);
  return ty;
}

function draw() {
  tf.tidy(() => {
    if (x_vals.length > 0) {
      // optimize
      const ty = tf.tensor1d(y_vals);
      opt.minimize(() => loss(predict(x_vals), ty));
    }
  });
  background(0);
  stroke(255);
  strokeWeight(6);
  for (let i = 0; i < x_vals.length; i++) {
    point(map(x_vals[i], 0, 1, 0, width), map(y_vals[i], 0, 1, height, 0));
  }
  strokeWeight(4);
  stroke(color(70, 230, 70));
  tf.tidy(() => {
    const xd = [0, 1];
    const yd = predict(xd);
    const yvals = yd.dataSync();
    let x1 = xd[0] * width;
    let y1 = map(yvals[0], 0, 1, height, 0);
    let x2 = xd[1] * width;
    let y2 = map(yvals[1], 0, 1, height, 0);
    line(x1, y1, x2, y2);
  });
  console.log(tf.memory().numTensors);
  fill(255);
  noStroke();
  text("Loss", 30, 25);
  text(loss_val.toFixed(6), 30, 40);
  text(`lr: ${lr.toFixed(2)}`, width - 60, 40);
  stroke(255);
  noFill();
  rect(0, 0, width, height);
  fill(100);
  point(mouseX, mouseY, 20);
}

function mousePressed() {
  if (mouseX > 0 && mouseX < width && mouseY > 0 && mouseY < height) {
    x_vals.push(map(mouseX, 0, width, 0, 1));
    y_vals.push(map(mouseY, 0, height, 1, 0));
  }
}

window.onresize = () => {
  resizeCanvas(windowWidth - 100, windowHeight - 100);
};
