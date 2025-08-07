for i in {1..6}; do

  echo "Running precision_calculator.py for run$i..."
  python3 precision_calculator.py $1 $2 $3 $i 1
  python3 precision_calculator.py $1 $2 $3 $i 2
  python3 precision_calculator.py $1 $2 $3 $i 3
done

for i in {7..12}; do


  echo "Running rms_error_calculator.py for run$i..."
  python3 rms_error_calculator.py $1 $2 $3 $i 1
  python3 rms_error_calculator.py $1 $2 $3 $i 2
  python3 rms_error_calculator.py $1 $2 $3 $i 3
done

for i in {13..14}; do

  echo "Running multiple_columns_precision_calculator.py for run$i..."
  python3 multiple_columns_precision_calculator.py $1 $2 $3 $i 1
  python3 multiple_columns_precision_calculator.py $1 $2 $3 $i 2
  python3 multiple_columns_precision_calculator.py $1 $2 $3 $i 3
done

for i in {16..21}; do

  echo "Running precision_calculator.py for run$i..."
  python3 precision_calculator.py $1 $2 $3 $i 1
  python3 precision_calculator.py $1 $2 $3 $i 2
  python3 precision_calculator.py $1 $2 $3 $i 3
done

for i in {22..23}; do


  echo "Running rms_error_calculator.py for run$i..."
  python3 rms_error_calculator.py $1 $2 $3 $i 1
  python3 rms_error_calculator.py $1 $2 $3 $i 2
  python3 rms_error_calculator.py $1 $2 $3 $i 3
done

for i in {24..25}; do

  echo "Running multiple_columns_precision_calculator.py for run$i..."
  python3 multiple_columns_precision_calculator.py $1 $2 $3 $i 1
  python3 multiple_columns_precision_calculator.py $1 $2 $3 $i 2
  python3 multiple_columns_precision_calculator.py $1 $2 $3 $i 3
done

for i in {26..31}; do

  echo "Running precision_calculator.py for run$i..."
  python3 precision_calculator.py $1 $2 $3 $i 1
  python3 precision_calculator.py $1 $2 $3 $i 2
  python3 precision_calculator.py $1 $2 $3 $i 3
done

for i in {32..33}; do


  echo "Running rms_error_calculator.py for run$i..."
  python3 rms_error_calculator.py $1 $2 $3 $i 1
  python3 rms_error_calculator.py $1 $2 $3 $i 2
  python3 rms_error_calculator.py $1 $2 $3 $i 3
done

for i in {34..35}; do

  echo "Running multiple_columns_precision_calculator.py for run$i..."
  python3 multiple_columns_precision_calculator.py $1 $2 $3 $i 1
  python3 multiple_columns_precision_calculator.py $1 $2 $3 $i 2
  python3 multiple_columns_precision_calculator.py $1 $2 $3 $i 3
done

for i in {36..39}; do

  echo "Running precision_calculator.py for run$i..."
  python3 precision_calculator.py $1 $2 $3 $i 1
  python3 precision_calculator.py $1 $2 $3 $i 2
  python3 precision_calculator.py $1 $2 $3 $i 3
done

