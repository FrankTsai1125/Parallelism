#!/bin/bash
# 準備提交檔案的腳本
# 使用方式：bash prepare_submission.sh <student_id>

if [ -z "$1" ]; then
    echo "Usage: bash prepare_submission.sh <student_id>"
    echo "Example: bash prepare_submission.sh b12345678"
    exit 1
fi

STUDENT_ID=$1
SUBMIT_DIR="${STUDENT_ID}"

echo "=== 準備提交檔案 ==="
echo "學號: $STUDENT_ID"
echo ""

# 檢查檔案
echo "檢查必要檔案..."
REQUIRED_FILES="hw2.cpp sift.cpp sift.hpp image.cpp image.hpp Makefile"
MISSING=""

for file in $REQUIRED_FILES; do
    if [ ! -f "$file" ]; then
        MISSING="$MISSING $file"
    else
        echo "  ✓ $file"
    fi
done

if [ -n "$MISSING" ]; then
    echo "❌ 缺少檔案:$MISSING"
    exit 1
fi

# 檢查 report.pdf
if [ ! -f "report.pdf" ]; then
    echo "  ⚠️  report.pdf (尚未準備)"
    echo ""
    echo "請記得撰寫報告並命名為 report.pdf"
    read -p "繼續製作提交檔案？(y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "  ✓ report.pdf"
fi

echo ""
echo "建立提交目錄..."

# 清理舊的
rm -rf "$SUBMIT_DIR" "${SUBMIT_DIR}.tar"

# 建立目錄
mkdir -p "$SUBMIT_DIR"

# 複製檔案
cp hw2.cpp sift.cpp sift.hpp image.cpp image.hpp Makefile "$SUBMIT_DIR/"
if [ -f "report.pdf" ]; then
    cp report.pdf "$SUBMIT_DIR/"
fi

# 建立 tar 檔案
tar cvf "${SUBMIT_DIR}.tar" "$SUBMIT_DIR"

echo ""
echo "=== 提交檔案已準備完成 ==="
echo "檔案名稱: ${SUBMIT_ID}.tar"
echo "檔案大小: $(du -h ${SUBMIT_DIR}.tar | cut -f1)"
echo ""
echo "內容："
tar -tvf "${SUBMIT_DIR}.tar"
echo ""
echo "✅ 請將 ${SUBMIT_DIR}.tar 上傳至 NTU COOL"
echo ""
