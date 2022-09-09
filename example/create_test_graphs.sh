data_dir=data
cache_dir=cache
destination_dir=graphs
structure_file=structure.yml

cd $(dirname $0)

download_pubmed_data.sh --source updatefiles \
    --destination $data_dir \
    {1357..1361}

if [ $DISAMBIGUATE_SMALL ]; then
    read_xml --structure-file $structure_file \
        --cache-dir $cache_dir \
        $data_dir/pubmed22n1360.xml.gz # 60 is smallest file downloaded.

    destination_dir=${destination_dir}_small
else
    read_xml --structure-file $structure_file \
        --cache-dir $cache_dir \
        $data_dir/*
fi

convert2graph.sh --structure-file $structure_file \
    --source $cache_dir \
    --destination $destination_dir \
    --clean
